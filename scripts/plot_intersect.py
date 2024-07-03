#!/usr/bin/env python

import rospy
import copy
import time

from uwb_tdoa.msg import TDoAAnchor, TDoAMeas
from dwm1001_ros.msg import UWBMeas
from xplraoa_ros.msg import Angles
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import numpy as np
from numpy.polynomial import Polynomial
import tf2_ros
import tf2_py as tf2
import ros_numpy
import matplotlib.pyplot as plt
import sys
import genpy
import threading
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfilter_zi, butter
from collections import deque, defaultdict

C = 299792458
CUTOFF = 1.5
RANGE = 4.0
CALIB_LEN = 50


def plot_tdoa(an1, an2, diff, x, y, ax: plt.Axes, var):
    dist = np.linalg.norm(an1 - an2)
    d1 = np.sqrt((x - an1[0]) ** 2 + (y - an1[1]) ** 2)
    d2 = np.sqrt((x - an2[0]) ** 2 + (y - an2[1]) ** 2)
    if dist > np.abs(diff):
        ax.contour(x, y, d1 - d2, [diff])
        # ax.contourf(x, y, d1 - d2, [diff - var, diff + var], alpha=0.1)
    else:
        if diff > 0:
            start = an2
            dir = an2 - an1
        else:
            start = an1
            dir = an1 - an2
        ax.plot(
            [start[0], start[0] + 1000 * dir[0]],
            [start[1], start[1] + 1000 * dir[1]],
            color="black",
        )
        # ax.contourf(
        #     x, y, d1 - d2, [diff - var, np.sign(diff) * (dist - 0.01)], alpha=0.3
        # )


def get_angle(Va, Vb, Vn):
    """returns oriented angle of rotation from Va to Vb"""
    # https://stackoverflow.com/a/33920320
    return float(
        np.arctan2(np.matmul(np.cross(Va, Vb, axis=0).T, Vn), np.matmul(Va.T, Vb))
    )


class Plotter:
    def __init__(self) -> None:
        rospy.init_node("plotter")
        self.msg_lock = threading.Lock()
        self.stamps = {}
        self.vars = {}
        self.frame_sn = {}

        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.2)

        # TDoA config
        self.anchors_address = rospy.get_param("anchors/address")
        self.coords = rospy.get_param("anchors/position")
        self.num_anchors = rospy.get_param("num_anchors")
        self.positions = {}
        self.tof = {}
        self.lp_filter = butter(3, CUTOFF, fs=10.0)
        self.zi = {}
        self.calib_queue = defaultdict(dict)
        self.calib = {}
        for i in range(len(self.anchors_address)):
            address = self.anchors_address[i]
            self.positions[address] = np.array(
                [
                    [self.coords[3 * i]],
                    [self.coords[3 * i + 1]],
                    [self.coords[3 * i + 2]],
                ]
            )
            self.zi[address] = lfilter_zi(self.lp_filter[0], self.lp_filter[1])
            self.calib[address] = Polynomial([0.0, 1.0])

        # TWR config
        self.target_anchor = rospy.get_param("target/anchor")
        self.target_tags = rospy.get_param("target/tags")
        self.target_ids = rospy.get_param("/uwb/twr/ids")
        self.target_coords = rospy.get_param("/uwb/twr/positions")
        self.target_positions = {}
        self.subscribers = {}
        for i in range(len(self.target_ids)):
            id_l = self.target_ids[i].lower()
            self.target_positions[id_l] = np.array(
                [
                    [self.target_coords[3 * i]],
                    [self.target_coords[3 * i + 1]],
                    [self.target_coords[3 * i + 2]],
                ]
            )

        # position estimation
        self.last_estimate = None
        self.tim2 = rospy.Timer(rospy.Duration(0.1), self.get_estimate)

        # angle measurements
        self.angle_zi = lfilter_zi(self.lp_filter[0], self.lp_filter[1])
        self.angle_pub = rospy.Publisher("angle", Angles, queue_size=1)

        self.anchors = list(self.positions.keys())

        self.marker_pub = rospy.Publisher("heading", Marker, queue_size=1)

        self.subs1 = rospy.Subscriber(
            "measurements", TDoAMeas, self.tdoa_cb, queue_size=1
        )

        self.tim = rospy.Timer(rospy.Duration(1.0), self.auto_calibrate)
        print("setup done")

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        self.tim.shutdown()
        rospy.sleep(0.2)

    def get_transform(self, tf_from, tf_to, out="matrix", time=None, dur=0.1):
        """returns the latest transformation between the given frames
        the result of multiplying point in frame tf_to by the output matrix is in the frame tf_from

        :param tf_from: find transform from this frame
        :param tf_to: find transform to this frame
        :param out: the return type
                    - 'matrix' - returns numpy array with the tf matrix
                    - 'tf' - returns TransformStamped
        :param time: the desired timestamp of the transform (ROS Time)
        :param dur: the timeout of the lookup (float)
        :return: as selected by out parameter or None in case of tf2 exception
                    - only ConnectivityException is logged
        """
        if time is None:
            tf_time = rospy.Time(0)
        else:
            if not isinstance(time, rospy.Time) and not isinstance(time, genpy.Time):
                raise TypeError("parameter time has to be ROS Time")
            tf_time = time

        try:
            t = self.tf_buffer.lookup_transform(
                tf_from, tf_to, tf_time, rospy.Duration(dur)
            )
        except (tf2.LookupException, tf2.ExtrapolationException):
            return None
        except tf2.ConnectivityException as ex:
            rospy.logerr(ex)
            return None

        # return the selected type
        if out == "matrix":
            return ros_numpy.numpify(t.transform)
        elif out == "tf":
            return t
        else:
            raise ValueError("argument out should be 'matrix' or 'tf'")

    def get_estimate(self, _):
        t = self.get_transform("locator", "position")
        if t is not None:
            est = np.array([t[0, 3], t[1, 3]])
            est /= np.linalg.norm(est)
            self.last_estimate = est

    def tdoa_cb(self, msg: TDoAMeas):
        self.msg_lock.acquire()
        tag = False
        stamps = {}
        for m in msg.measurements:
            if m.target == self.target_anchor:
                if m.meas_valid:
                    self.stamps[m.address] = m.stamp
                    self.vars[m.address] = m.variance
                    self.frame_sn[m.address] = m.frame_sn
            else:
                if m.meas_valid:
                    tag = True
                    stamps[m.address] = m.stamp
        self.msg_lock.release()
        if tag:
            for i in range(len(self.anchors)):
                id1 = self.anchors[i]
                id2 = self.anchors[(i + 1) % len(self.anchors)]
                d1 = np.linalg.norm(
                    self.positions[id1] - self.target_positions[m.target]
                )
                d2 = np.linalg.norm(
                    self.positions[id2] - self.target_positions[m.target]
                )
                expected_diff = d1 - d2
                diff = C * (stamps[id1] - stamps[id2])
                if m.target not in self.calib_queue[id1].keys():
                    self.calib_queue[id1][m.target] = deque()
                self.calib_queue[id1][m.target].append((expected_diff, diff))

    def auto_calibrate(self, _):
        for a in self.anchors:
            l = []
            for t in self.target_tags:
                if t not in self.calib_queue[a].keys():
                    continue
                while len(self.calib_queue[a][t]) >= CALIB_LEN:
                    self.calib_queue[a][t].popleft()
                if len(self.calib_queue[a][t]) != 0:
                    mean = np.mean(self.calib_queue[a][t], axis=0)
                    l += [(mean[1], mean[0])]
            l = np.array(l)
            if len(l) == 0:
                return
            elif len(l) == 1:
                c = np.mean(l[:, 0] - l[:, 1])
                self.calib[a] = Polynomial([c, 1.0])
            else:
                p = Polynomial.fit(l[:, 1], l[:, 0], l.shape[0] - 1)
                p = p.convert()
                self.calib[a] = p

    def intersectionPoint(self, guess, p_tdoa, d_tdoa, w):
        gu = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 1]), np.array([0, -1])]

        # TDoA
        x_t1 = []
        y_t1 = []
        x_t2 = []
        y_t2 = []
        for i in range(len(p_tdoa)):
            x_t1 += [[p_tdoa[i][0][0, 0]]]
            y_t1 += [[p_tdoa[i][0][1, 0]]]
            x_t2 += [[p_tdoa[i][1][0, 0]]]
            y_t2 += [[p_tdoa[i][1][1, 0]]]
        x_t1_tdoa = np.array(x_t1)
        y_t1_tdoa = np.array(y_t1)
        x_t2_tdoa = np.array(x_t2)
        y_t2_tdoa = np.array(y_t2)
        d_tdoa = np.array(d_tdoa)

        last = self.last_estimate

        def eq(g):
            x, y = g

            f_tdoa = (
                np.sqrt((x - x_t1_tdoa) ** 2 + (y - y_t1_tdoa) ** 2)
                - np.sqrt((x - x_t2_tdoa) ** 2 + (y - y_t2_tdoa) ** 2)
                - d_tdoa
            ) ** 2
            # f_tdoa = w * f_tdoa
            f = np.vstack((f_tdoa, 10 * (np.linalg.norm(g) - 1)))
            if last is not None:
                f = np.vstack((f, 0.5 * np.linalg.norm(g - last)))
            return f.flatten().tolist()

        best = None
        cost = np.inf
        for g in gu:
            ans = least_squares(eq, g, loss="soft_l1", verbose=0)
            if ans.success and ans.cost < cost:
                best = ans.x
                cost = ans.cost
        return best

        if ans.success:
            return ans.x
        else:
            return None


if __name__ == "__main__":
    arg = rospy.myargv(argv=sys.argv)
    p = Plotter()

    x = np.linspace(-RANGE, RANGE, 1000)
    y = np.linspace(-RANGE, RANGE, 1000)
    x, y = np.meshgrid(x, y)

    fig, ax1 = plt.subplots(1, 1)
    plt.ion()

    # angles = []
    marker = Marker()
    marker.header.frame_id = "locator"
    marker.type = marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.1
    marker.scale.y = 0.2
    marker.scale.z = 0.3
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0

    while not rospy.is_shutdown():
        ax1.cla()
        if len(p.stamps.keys()) != p.num_anchors:
            rospy.logwarn("No data")
            rospy.sleep(0.1)
            continue

        p.msg_lock.acquire()
        stamps = copy.deepcopy(p.stamps)
        p.msg_lock.release()

        # plot frame
        ax1.quiver(0, 0, 1, 0, color="r", scale=5)
        ax1.quiver(0, 0, 0, 1, color="g", scale=5)

        d = []
        pos = []
        w = []
        for i in range(len(p.anchors)):
            id1 = p.anchors[i]
            id2 = p.anchors[(i + 1) % len(p.anchors)]
            a1 = p.positions[id1]
            a2 = p.positions[id2]
            d_a2a = np.linalg.norm(a1 - a2)
            # diff = p.calib[id1](C * (stamps[id1] - stamps[id2]))
            diff = C * (stamps[id1] - stamps[id2])

            if diff > d_a2a:
                diff = d_a2a

            w += [[1 - np.abs(diff) / d_a2a]]

            var = C * (p.vars[id1] + p.vars[id2])
            diff_avg, p.zi[id1] = lfilter(
                p.lp_filter[0], p.lp_filter[1], np.array([diff]), zi=p.zi[id1]
            )
            diff_avg = float(diff_avg)

            d += [[diff_avg]]
            pos += [[a1, a2]]

            plot_tdoa(a1, a2, diff_avg, x, y, ax1, var)

        # solve NLS
        if p.last_estimate is None:
            guess = np.array([1.0, 0.0])
        else:
            guess = p.last_estimate
        est = p.intersectionPoint(guess, pos, d, w)
        if est is None:
            rospy.logwarn("No solution")
        else:
            Va = np.array([[1.0], [0.0], [0.0]])
            Vb = np.array([[est[0]], [est[1]], [0.0]])
            Vn = np.array([[0.0], [0.0], [1.0]])
            angle = get_angle(Va, Vb, Vn)
            # angles += [angle]
            # angle_f, p.angle_zi = lfilter(
            #     p.lp_filter[0], p.lp_filter[1], np.array([angle]), zi=p.angle_zi
            # )
            ax1.plot(
                [0, 1000 * np.cos(angle)],
                [0, 1000 * np.sin(angle)],
                linewidth=4,
                color="r",
            )
            ax1.scatter(est[0], est[1], s=500, marker="x", zorder=4)
            msg = Angles(angle, 0.0, 0.0)
            p.angle_pub.publish(msg)
            marker.header.stamp = rospy.Time.now()
            marker.points = [
                Point(0, 0, 0),
                Point(2 * np.cos(angle), 2 * np.sin(angle), 0),
            ]
            p.marker_pub.publish(marker)

        # plot TWR
        # if p.range is not None:
        #     phis = np.arange(0, 2 * np.pi, 0.01)
        #     ax1.plot(p.range * np.cos(phis), p.range * np.sin(phis))

        # plot anchors
        for i in range(len(p.anchors)):
            id1 = p.anchors[i]
            ax1.scatter(p.positions[id1][0], p.positions[id1][1], s=100, zorder=4)
        ax1.set_xlim(-RANGE, RANGE)
        ax1.set_ylim(-RANGE, RANGE)
        ax1.grid(True)
        ax1.set_aspect("equal")
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)
        rospy.sleep(0.1)
