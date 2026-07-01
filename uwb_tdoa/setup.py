import os
from glob import glob
from setuptools import find_packages, setup

package_name = "uwb_tdoa"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bohumil Brož",
    maintainer_email="brozbohu@fel.cvut.cz",
    description="Driver and locator for UWB TDoA anchors",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "driver = uwb_tdoa.driver:main",
            "tdoa_locator = uwb_tdoa.tdoa_locator:main",
            "plot_intersect = uwb_tdoa.plot_intersect:main",
        ],
    },
)
