import math
import threading
import time

import av
import av.container

import numpy as np
from tellopy._internal import error
from tellopy._internal import event
from tellopy._internal import logger
from tellopy._internal import protocol
from tellopy._internal import tello

import builtin_interfaces.msg
import rclpy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from h264_msgs.msg import Packet
from tello_interfaces.msg import TelloStatus
from sensor_msgs.msg import CameraInfo


def yaml_to_camerainfo(calib_yaml: str) -> CameraInfo:
    """
    Parse a yaml file containing camera calibration data (as produced by
    rosrun camera_calibration cameracalibrator.py) into a
    sensor_msgs/CameraInfo msg.
    Parameters
    ----------
    calib_yaml : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    calib_data = yaml.load(open(calib_yaml).read(), Loader=yaml.FullLoader)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.k = calib_data["camera_matrix"]["data"]
    camera_info_msg.d = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.r = calib_data["rectification_matrix"]["data"]
    camera_info_msg.p = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg


class TelloNode(Node):

    def __init__(self):
        super().__init__('tello_node')
        self.declare_parameter('camera_calibration', '/home/lio/tcc/tello_ws/src/tello_driver/cfg/960x720.yaml')
        self.declare_parameter('h264', True)
        self.declare_parameter('zoom', False)
        self.declare_parameter('fast_mode', False)
        self.declare_parameter('vel_scale', 1)

        self.stream_h264_video = self.get_parameter('h264').get_parameter_value().bool_value
        self.zoom = self.get_parameter('zoom').get_parameter_value().bool_value
        self.vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        self.fast_mode = self.get_parameter("fast_mode").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.frame_thread = None

        # Connect to drone
        self.logger = self.get_logger()

        self.tello = tello.Tello()
        self.tello.set_loglevel(logger.LOG_INFO)
        self.logger.info('Connecting to drone @ %s:%d' % self.tello.tello_addr)
        self.tello.connect()
        try:
            self.tello.wait_for_connection(timeout=10)
        except error.TelloError as err:
            self.logger.error(str(err))
            self.destroy_node()
            rclpy.shutdown()
            return
        self.logger.info('Connected to drone')
        # rclpy.on_shutdown(self.cb_shutdown)

        self.pub_status = self.create_publisher(TelloStatus, 'status', 1)
        if self.stream_h264_video:
            self.pub_image_h264 = self.create_publisher(
                Packet, 'image_raw/h264', 1)
        else:
            self.pub_image_raw = self.create_publisher(
                Image, 'camera/image_raw', 1)

        # self.sub_takeoff = rclpy.Subscriber('takeoff', Empty, self.cb_takeoff)
        # self.sub_manual_takeoff = rclpy.Subscriber(
        #     'manual_takeoff', Empty, self.cb_manual_takeoff)
        # self.sub_throw_takeoff = rclpy.Subscriber(
        #     'throw_takeoff', Empty, self.cb_throw_takeoff)
        # self.sub_land = rclpy.Subscriber('land', Empty, self.cb_land)
        # self.sub_palm_land = rclpy.Subscriber(
        #     'palm_land', Empty, self.cb_palm_land)
        # self.sub_flattrim = rclpy.Subscriber(
        #     'flattrim', Empty, self.cb_flattrim)
        # self.sub_flip = rclpy.Subscriber('flip', UInt8, self.cb_flip)
        self.sub_cmd_vel = self.create_subscription(Twist, 'cmd_vel', self.cb_cmd_vel, 1)
        # self.sub_fast_mode = rclpy.Subscriber(
        #     'fast_mode', Empty, self.cb_fast_mode)

        self.tello.subscribe(self.tello.EVENT_FLIGHT_DATA, self.cb_status_log)

        # Reconstruction H264 video frames
        self.prev_seq_id = None
        self.seq_block_count = 0

        # Height from EVENT_FLIGHT_DATA more accurate than MVO (monocular visual odometry)
        self.height = 0

        # EVENT_LOG_DATA from 'TelloPy' package
        self.pub_odom = self.create_publisher(Odometry, 'odom', 1)
        self.pub_imu = self.create_publisher(Imu, 'imu', 1)

        self.tello.subscribe(self.tello.EVENT_LOG_DATA, self.cb_data_log)

        self.tello.EVENT_VIDEO_FRAME_H264 = event.Event('video frame h264')
        if self.stream_h264_video:
            self.tello.start_video()
            self.tello.subscribe(self.tello.EVENT_VIDEO_DATA, self.cb_video_data)
            self.tello.subscribe(self.tello.EVENT_VIDEO_FRAME_H264, self.cb_h264_frame)
            pass
        else:
            self.frame_thread = threading.Thread(target=self.framegrabber_loop)
            self.frame_thread.start()

        calib_path = self.get_parameter('camera_calibration').get_parameter_value().string_value
        self.caminfo = yaml_to_camerainfo(calib_path)
        self.caminfo.header.frame_id = "camera_link"
        self.pub_caminfo = self.create_publisher(CameraInfo, 'camera/camera_info', 1)
        self.pub_caminfo.publish(self.caminfo)

        self.left_x = 0.
        self.left_y = 0.
        self.right_x = 0.
        self.right_y = 0.

        self.logger.info('Tello driver node ready')

    def set_fast_mode(self, enabled):
        self.fast_mode = enabled

    def reset_cmd_vel(self):
        self.left_x = 0.
        self.left_y = 0.
        self.right_x = 0.
        self.right_y = 0.
        self.fast_mode = False

    # scaling for velocity command
    def __scale_vel_cmd(self, cmd_val):
        return self.vel_scale * cmd_val

    def __send_stick_command(self):
        pkt = protocol.Packet(protocol.STICK_CMD, 0x60)

        axis1 = int(1024 + 660.0 * self.right_x) & 0x7ff
        axis2 = int(1024 + 660.0 * self.right_y) & 0x7ff
        axis3 = int(1024 + 660.0 * self.left_y) & 0x7ff
        axis4 = int(1024 + 660.0 * self.left_x) & 0x7ff
        axis5 = int(self.fast_mode) & 0x01
        self.logger.debug("stick command: fast=%d yaw=%4d vrt=%4d pit=%4d rol=%4d" %
                          (axis5, axis4, axis3, axis2, axis1))

        '''
        11 bits (-1024 ~ +1023) x 4 axis = 44 bits
        fast_mode takes 1 bit
        44+1 bits will be packed in to 6 bytes (48 bits)
         axis5      axis4      axis3      axis2      axis1
             |          |          |          |          |
                 4         3         2         1         0
        98765432109876543210987654321098765432109876543210
         |       |       |       |       |       |       |
             byte5   byte4   byte3   byte2   byte1   byte0
        '''
        packed = axis1 | (axis2 << 11) | (
                axis3 << 22) | (axis4 << 33) | (axis5 << 44)
        packed_bytes = protocol.struct.pack('<Q', packed)
        pkt.add_byte(protocol.byte(packed_bytes[0]))
        pkt.add_byte(protocol.byte(packed_bytes[1]))
        pkt.add_byte(protocol.byte(packed_bytes[2]))
        pkt.add_byte(protocol.byte(packed_bytes[3]))
        pkt.add_byte(protocol.byte(packed_bytes[4]))
        pkt.add_byte(protocol.byte(packed_bytes[5]))
        pkt.add_time()
        pkt.fixup()
        self.logger.debug("stick command: %s" %
                          protocol.byte_to_hexstring(pkt.get_buffer()))
        return self.tello.send_packet(pkt)

    def manual_takeoff(self):
        # Hold max 'yaw' and min 'pitch', 'roll', 'throttle' for several seconds
        self.tello.set_pitch(-1)
        self.tello.set_roll(-1)
        self.tello.set_yaw(1)
        self.tello.set_throttle(-1)
        self.fast_mode = False

        return self.__send_stick_command()

    def cb_video_data(self, event, sender, data, **args):
        now = time.time()

        # parse packet
        seq_id = protocol.byte(data[0])
        sub_id = protocol.byte(data[1])
        packet = data[2:]
        self.tello.sub_last = False
        if sub_id >= 128:  # MSB asserted
            sub_id -= 128
            self.tello.sub_last = True

        # associate packet to (new) frame
        if self.prev_seq_id is None or self.prev_seq_id != seq_id:
            # detect wrap-arounds
            if self.prev_seq_id is not None and self.prev_seq_id > seq_id:
                self.seq_block_count += 1
            self.tello.frame_pkts = [None] * 128  # since sub_id uses 7 bits
            self.tello.frame_t = now
            self.prev_seq_id = seq_id
        self.tello.frame_pkts[sub_id] = packet

        # publish frame if completed
        if self.tello.sub_last and all(self.tello.frame_pkts[:sub_id + 1]):
            if isinstance(self.tello.frame_pkts[sub_id], str):
                frame = ''.join(self.tello.frame_pkts[:sub_id + 1])
            else:
                frame = b''.join(self.tello.frame_pkts[:sub_id + 1])
            self.tello._Tello__publish(event=self.tello.EVENT_VIDEO_FRAME_H264,
                                       data=(frame, self.seq_block_count * 256 + seq_id, self.tello.frame_t))

    def send_req_video_sps_pps(self):
        """Manually request drone to send an I-frame info (SPS/PPS) for video stream."""
        pkt = protocol.Packet(protocol.VIDEO_START_CMD, 0x60)
        pkt.fixup()
        return self.tello.send_packet(pkt)

    def set_video_req_sps_hz(self, hz):
        """Internally sends a SPS/PPS request at desired rate; <0: disable."""
        if hz < 0:
            hz = 0.
        self.tello.video_req_sps_hz = hz

    # emergency command
    def emergency(self):
        """ Stop all motors """
        self.logger.info('emergency (cmd=% seq=0x%04x)' %
                         (protocol.EMERGENCY_CMD, self.tello.pkt_seq_num))
        pkt = protocol.Packet(protocol.EMERGENCY_CMD)
        return self.tello.send_packet(pkt)

    def flip(self, cmd):
        """ tell drone to perform a flip in directions [0,8] """
        self.logger.info('flip (cmd=0x%02x seq=0x%04x)' %
                         (protocol.FLIP_CMD, self.tello.pkt_seq_num))
        pkt = protocol.Packet(protocol.FLIP_CMD, 0x70)
        pkt.add_byte(cmd)
        pkt.fixup()
        return self.tello.send_packet(pkt)

    def cb_video_mode(self, msg):
        if not self.zoom:
            self.tello.set_video_mode(True)
        else:
            self.tello.set_video_mode(False)

    # def cb_emergency(self, msg):
    #     success = self.emergency()
    #     notify_cmd_success('Emergency', success)

    def cb_status_log(self, event, sender, data: protocol.FlightData, **args):
        speed_horizontal_mps = math.sqrt(
            data.north_speed * data.north_speed + data.east_speed * data.east_speed) / 10.

        # TODO: verify outdoors: anecdotally, observed that:
        # data.east_speed points to South
        # data.north_speed points to East
        self.height = data.height / 10.
        msg = TelloStatus(
            height_m=data.height / 10.,
            speed_northing_mps=-data.east_speed / 10.,
            speed_easting_mps=data.north_speed / 10.,
            speed_horizontal_mps=speed_horizontal_mps,
            speed_vertical_mps=-data.ground_speed / 10.,
            flight_time_sec=data.fly_time / 10.,
            imu_state=bool(data.imu_state),
            pressure_state=bool(data.pressure_state),
            down_visual_state=bool(data.down_visual_state),
            power_state=bool(data.power_state),
            battery_state=bool(data.battery_state),
            gravity_state=bool(data.gravity_state),
            wind_state=bool(data.wind_state),
            imu_calibration_state=data.imu_calibration_state,
            battery_percentage=data.battery_percentage,
            drone_fly_time_left_sec=data.drone_fly_time_left / 10.,
            drone_battery_left_sec=data.drone_battery_left / 10.,
            is_flying=bool(data.em_sky),
            is_on_ground=bool(data.em_ground),
            is_em_open=bool(data.em_open),
            is_drone_hover=bool(data.drone_hover),
            is_outage_recording=bool(data.outage_recording),
            is_battery_low=bool(data.battery_low),
            is_battery_lower=bool(data.battery_lower),
            is_factory_mode=bool(data.factory_mode),
            fly_mode=data.fly_mode,
            throw_takeoff_timer_sec=data.throw_fly_timer / 10.,
            camera_state=data.camera_state,
            electrical_machinery_state=data.electrical_machinery_state,
            front_in=bool(data.front_in),
            front_out=bool(data.front_out),
            front_lsc=bool(data.front_lsc),
            temperature_height_m=data.temperature_height / 10.,
            cmd_roll_ratio=self.right_x,
            cmd_pitch_ratio=self.right_y,
            cmd_yaw_ratio=self.left_x,
            cmd_vspeed_ratio=self.left_y,
            cmd_fast_mode=bool(self.fast_mode),
        )
        self.pub_status.publish(msg)

    def cb_data_log(self, event, sender, data, **args):
        time_cb = self.get_clock().now()

        odom_msg = Odometry()
        odom_msg.child_frame_id = 'base_link'
        odom_msg.header.stamp = time_cb.to_msg()
        odom_msg.header.frame_id = 'odom'

        # Height from MVO received as negative distance to floor
        odom_msg.pose.pose.position.z = -data.mvo.pos_z  # self.height #-data.mvo.pos_z
        odom_msg.pose.pose.position.x = data.mvo.pos_x
        odom_msg.pose.pose.position.y = data.mvo.pos_y
        odom_msg.pose.pose.orientation.w = data.imu.q0
        odom_msg.pose.pose.orientation.x = data.imu.q1
        odom_msg.pose.pose.orientation.y = data.imu.q2
        odom_msg.pose.pose.orientation.z = data.imu.q3
        # Linear speeds from MVO received in dm/sec
        odom_msg.twist.twist.linear.x = data.mvo.vel_y / 10
        odom_msg.twist.twist.linear.y = data.mvo.vel_x / 10
        odom_msg.twist.twist.linear.z = -data.mvo.vel_z / 10
        odom_msg.twist.twist.angular.x = data.imu.gyro_x
        odom_msg.twist.twist.angular.y = data.imu.gyro_y
        odom_msg.twist.twist.angular.z = data.imu.gyro_z

        self.pub_odom.publish(odom_msg)

        imu_msg = Imu()
        imu_msg.header.stamp = time_cb.to_msg()
        imu_msg.header.frame_id = 'base_link'

        imu_msg.orientation.w = data.imu.q0
        imu_msg.orientation.x = data.imu.q1
        imu_msg.orientation.y = data.imu.q2
        imu_msg.orientation.z = data.imu.q3
        imu_msg.angular_velocity.x = data.imu.gyro_x
        imu_msg.angular_velocity.y = data.imu.gyro_y
        imu_msg.angular_velocity.z = data.imu.gyro_z
        imu_msg.linear_acceleration.x = data.imu.acc_x
        imu_msg.linear_acceleration.y = data.imu.acc_y
        imu_msg.linear_acceleration.z = data.imu.acc_z

        self.pub_imu.publish(imu_msg)

    def cb_cmd_vel(self, msg):
        self.tello.set_pitch(self.__scale_vel_cmd(msg.linear.y))
        self.tello.set_roll(self.__scale_vel_cmd(msg.linear.x))
        self.tello.set_yaw(self.__scale_vel_cmd(msg.angular.z))
        self.tello.set_throttle(self.__scale_vel_cmd(msg.linear.z))

    def cb_flip(self, msg):
        if msg.data < 0 or msg.data > 7:  # flip integers between [0,7]
            self.logger.warn('Invalid flip direction: %d' % msg.data)
            return
        success = self.flip(msg.data)
        # notify_cmd_success('Flip %d' % msg.data, success)

    # def cb_shutdown(self):
    #     self.quit()
    #     if self.frame_thread is not None:
    #         self.frame_thread.join()

    def cb_h264_frame(self, event, sender, data, **args):
        frame, seq_id, frame_secs = data
        pkt_msg = Packet()
        # pkt_msg.header.seq = seq_id
        pkt_msg.seq = seq_id
        pkt_msg.header.frame_id = 'camera_link'
        pkt_msg.header.stamp = builtin_interfaces.msg.Time(sec=int(frame_secs))
        pkt_msg.data = frame
        self.pub_image_h264.publish(pkt_msg)
        #
        # self.caminfo.header. = seq_id
        self.caminfo.header.stamp = builtin_interfaces.msg.Time(sec=int(frame_secs))
        self.pub_caminfo.publish(self.caminfo)

    def framegrabber_loop(self):
        # Repeatedly try to connect
        vs = self.tello.get_video_stream()
        while self.tello.state != self.tello.STATE_QUIT:
            try:
                container = av.open(vs)
                break
            except BaseException as err:
                self.logger.error('fgrab: pyav stream failed - %s' % str(err))
                time.sleep(1.0)

        # Once connected, process frames till drone/stream closes
        while self.tello.state != self.tello.STATE_QUIT:
            try:
                # vs blocks, dies on self.stop
                for frame in container.decode(video=0):
                    img = np.array(frame.to_image())
                    try:
                        img_msg = self.bridge.cv2_to_imgmsg(img, 'rgb8')
                        img_msg.header.frame_id = 'camera_link'
                    except CvBridgeError as err:
                        self.logger.error(
                            'fgrab: cv bridge failed - %s' % str(err))
                        continue
                    self.pub_image_raw.publish(img_msg)
                    self.pub_caminfo.publish(self.caminfo)
                break
            except BaseException as err:
                self.logger.error('fgrab: pyav decoder failed - %s' % str(err))

    # def cb_takeoff(self, msg):
    #     success = self.tello.takeoff()
    #     notify_cmd_success('Takeoff', success)
    #
    # def cb_manual_takeoff(self, msg):
    #     success = self.manual_takeoff()
    #     notify_cmd_success('Manual takeoff', success)
    #
    # def cb_throw_takeoff(self, msg):
    #     success = self.throw_and_go()
    #     if success:
    #         self.logger.info('Drone set to auto-takeoff when thrown')
    #     else:
    #         rclpy.logwarn('ThrowTakeoff command failed')
    #
    # def cb_land(self, msg):
    #     success = self.land()
    #     notify_cmd_success('Land', success)
    #
    # def cb_palm_land(self, msg):
    #     success = self.palm_land()
    #     notify_cmd_success('PalmLand', success)
    #
    # def cb_flattrim(self, msg):
    #     success = self.flattrim()
    #     notify_cmd_success('FlatTrim', success)

    def cb_fast_mode(self, msg):
        if self.fast_mode:
            self.set_fast_mode(False)
        elif not self.fast_mode:
            self.set_fast_mode(True)


def main(args=None):
    rclpy.init(args=args)
    node = TelloNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
