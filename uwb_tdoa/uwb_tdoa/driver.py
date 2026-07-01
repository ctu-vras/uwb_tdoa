#!/usr/bin/env python3

import socket
import errno
import threading
from collections import defaultdict

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from uwb_tdoa_interfaces.msg import TDoAMeas, TDoAAnchor
from std_msgs.msg import String

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
QUEUE_SIZE = 20

C = 299792458
FREQ = 63.8976 * 10**9  # GHz

DEBUG = False


class UDPPacket:
    def __init__(self, packet: bytes) -> None:
        self.raw = packet
        self.counter = 0

    def set_position(self, string):
        "set cursor behind the provided string"
        i = self.raw.find(string)
        if i == -1:
            return False
        self.counter += i + len(string)
        return True

    def read(self, size):
        data = self.raw[self.counter : self.counter + size]
        self.counter += size
        return data


class Driver(Node):
    def __init__(self) -> None:
        super().__init__("tdoa_driver")

        self.sound_pub = self.create_publisher(String, "/log_sound", 1)

        self.declare_parameter("server.address", Parameter.Type.STRING)
        self.declare_parameter("server.port", Parameter.Type.INTEGER)
        self.declare_parameter("num_anchors", Parameter.Type.INTEGER)
        self.declare_parameter("master", Parameter.Type.STRING)
        self.declare_parameter("anchors.address", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("anchors.sn", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("anchors.position", Parameter.Type.DOUBLE_ARRAY)

        self.ip = self.get_parameter("server.address").value
        self.port = self.get_parameter("server.port").value
        self.num_anchors = self.get_parameter("num_anchors").value
        self.master_address = self.get_parameter("master").value
        self.anchors_address = list(self.get_parameter("anchors.address").value)
        self.anchors_sn = list(self.get_parameter("anchors.sn").value)
        self.coords = list(self.get_parameter("anchors.position").value)

        self.positions = {}
        self.tof = {}
        self.last_received = {}
        self.friendly_name = {}
        for i in range(len(self.anchors_address)):
            address = self.anchors_address[i]
            self.positions[address] = np.array(
                [
                    [self.coords[3 * i]],
                    [self.coords[3 * i + 1]],
                    [self.coords[3 * i + 2]],
                ]
            )
            self.last_received[address] = None
            self.friendly_name[address] = self.anchors_sn[i]
        m = self.positions[self.master_address]
        for a in self.anchors_address:
            self.tof[a] = np.linalg.norm(self.positions[a] - m) / C

        self.last_send = -1
        self.message_queue = defaultdict(dict)
        self.queue_lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet  # UDP
        try:
            self.sock.bind((self.ip, self.port))
        except OSError as exc:
            if exc.errno == errno.EADDRNOTAVAIL:
                self.get_logger().fatal(
                    "OSError: [Errno 99] Cannot assign requested address [%s]" % self.ip
                )
                self.log_sound("T D O A: O S Error: Cannot assign requested address")
                raise RuntimeError("Cannot open the socket")
            else:
                raise

        self.sock.settimeout(0.3)

        self.prf_mask = 0xF0
        self.br_mask = 0x0F

        self.pub = self.create_publisher(TDoAMeas, "measurements", 1)

        self.tim = self.create_timer(0.01, self.send_meas)
        self.tim2 = self.create_timer(2.0, self.check_alive)

        # blocking receive loop runs in its own thread so the timers keep firing
        self.rx_thread = threading.Thread(target=self.run, daemon=True)
        self.rx_thread.start()

    def log_sound(self, text):
        self.sound_pub.publish(String(data=text))

    def shutdown(self):
        self.get_logger().info("Quitting...")
        self.tim.cancel()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError as exc:
            if exc.errno == errno.ENOTCONN:
                pass  # Socket is not connected, so can't send FIN packet.
            else:
                raise
        self.sock.close()

    def run(self):
        """decode the UDP packet"""
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                if rclpy.ok():
                    self.get_logger().error(
                        "Communication timeout", throttle_duration_sec=2.0
                    )
                    self.log_sound("T D O A: Communication timeout")
                continue
            except OSError:
                # socket closed during shutdown
                break
            packet = UDPPacket(data)
            if not packet.set_position(MAGIC):
                continue
            address = packet.read(8)[::-1].hex()
            platform = packet.read(2)[::-1].hex()
            version = packet.read(2)[::-1].hex()
            length = int.from_bytes(packet.read(2), "little")
            frame_sn = int.from_bytes(packet.read(1), "little")
            tlv_count = int.from_bytes(packet.read(1), "little")
            hdr_crc = packet.read(2)[::-1].hex()
            frame_crc = packet.read(2)[::-1].hex()
            if DEBUG:
                print("from:", addr)
                print("address:", address)
                print("platform:", platform)
                print("version:", version)
                print("length:", length)
                print("frame_sn:", frame_sn)
                print("tlv_count:", tlv_count)
                print("hdr_crc:", hdr_crc)
                print("frame_crc:", frame_crc)
                print()

            tlvs = []
            target = ""
            for i in range(tlv_count):
                type_b = packet.read(4)
                type = int.from_bytes(type_b[:2], "little")
                tlv_length = int.from_bytes(packet.read(4), "little")
                tlv_fields = []
                if type == 1:
                    mac = packet.read(tlv_length)[::-1].hex()
                    target = mac[6:10]
                    tlv_fields = [mac]
                elif type == 2:
                    payload = packet.read(tlv_length)
                    tlv_fields = [payload[::-1].hex()]
                elif type == 3:
                    rcv_time = int.from_bytes(packet.read(8), "little")
                    # move from ticks to seconds
                    rcv_time = rcv_time / FREQ
                    rcv_var = int.from_bytes(packet.read(8), "little")
                    rcv_var = rcv_var / FREQ
                    tlv_fields = [rcv_time, rcv_var]
                elif type == 4:
                    rssi = -(65535 - int.from_bytes(packet.read(2), "little")) / 100
                    fpp = -(65535 - int.from_bytes(packet.read(2), "little")) / 100
                    mc = int.from_bytes(packet.read(2), "little")
                    temp = int.from_bytes(packet.read(2), "little") / 10
                    tlv_fields = [rssi, fpp, mc, temp]
                elif type == 5:
                    rtto = int.from_bytes(packet.read(3), "little")
                    integrator = int.from_bytes(packet.read(3), "little")
                    channel = int.from_bytes(packet.read(1), "little")
                    prf_br = int.from_bytes(packet.read(1), "little")
                    prf = (prf_br & self.prf_mask) >> 4
                    br = prf_br & self.br_mask
                    tlv_fields = [rtto, integrator, channel, prf, br]
                else:
                    packet.read(tlv_length)
                tlvs += [(type, tlv_fields)]
            an = TDoAAnchor()
            an.ip_address = addr[0]
            an.address = address
            an.frame_sn = frame_sn
            an.target = target
            an.meas_valid = False
            an.type4_received = False
            an.type5_received = False
            for t in tlvs:
                if t[0] == 1:
                    an.mac = t[1][0]
                elif t[0] == 2:
                    an.payload = t[1][0]
                elif t[0] == 3:
                    an.meas_valid = True
                    an.stamp = rcv_time + self.tof[address]
                    an.variance = rcv_var
                elif t[0] == 4:
                    an.type4_received = True
                    an.rssi = float(rssi)
                    an.fpp = float(fpp)
                    an.los = float(mc)
                    an.temperature = float(temp)
                elif t[0] == 5:
                    an.type5_received = True
                    an.rtto = rtto
                    an.integrator = integrator
                    an.channel = channel
                    an.prf = prf
                    an.br = br
            id = float(np.round(an.stamp, 3))
            self.queue_lock.acquire()
            if id in self.message_queue[target].keys():
                self.message_queue[target][id] += [an]
            else:
                self.message_queue[target][id] = [an]
            self.queue_lock.release()
            self.last_received[address] = self.get_clock().now()

    def send_meas(self):
        """find the latest new complete measurement and publish it"""
        self.queue_lock.acquire()
        targets = self.message_queue.keys()
        for t in targets:
            keys = list(self.message_queue[t].keys())
            if len(keys) == 0:
                continue
            stamps = np.sort(np.array(keys))[::-1]
            if len(stamps) > QUEUE_SIZE:
                self.get_logger().warn(
                    "A lot of incomplete TDoA measurements, are all anchors connected?",
                    throttle_duration_sec=3.0,
                )
                self.log_sound(
                    "A lot of incomplete T D O A measurements, are all anchors connected?"
                )
                for i in range(QUEUE_SIZE, len(stamps)):
                    self.message_queue[t].pop(stamps[i])
                stamps = stamps[:QUEUE_SIZE]
            i = 0
            while True:
                meas_list = self.message_queue[t][stamps[i]]
                if len(meas_list) == self.num_anchors:
                    # ready to be sent
                    if stamps[i] > self.last_send:
                        m = TDoAMeas()
                        m.stamp = self.get_clock().now().to_msg()
                        m.measurements = meas_list
                        self.pub.publish(m)
                        self.last_send = stamps[i]
                        self.message_queue[t].pop(stamps[i])
                    else:
                        break
                i += 1
                if i == len(stamps):
                    # nothing to send
                    break
            keys = list(self.message_queue[t].keys())
            for k in keys:
                if k < self.last_send:
                    self.message_queue[t].pop(k)
        self.queue_lock.release()

    def check_alive(self):
        """check if all anchors are connected and sending measurements"""
        tim = self.get_clock().now()
        for a in self.anchors_address:
            if self.last_received[a] is None:
                self.get_logger().warn(
                    "No data yet from anchor with SN %s" % self.friendly_name[a],
                    throttle_duration_sec=3.0,
                )
                self.log_sound(
                    "T D O A: No data yet from anchor with SN %s" % self.friendly_name[a]
                )
            elif tim - self.last_received[a] > Duration(seconds=3.0):
                self.get_logger().error(
                    "No data received for more than 3 seconds from anchor with SN %s"
                    % self.friendly_name[a],
                )
                self.log_sound(
                    "T D O A: No data received for more than 3 seconds from anchor with SN %s"
                    % self.friendly_name[a]
                )
                self.last_send = -1


def main(args=None):
    rclpy.init(args=args)

    node = Driver()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
