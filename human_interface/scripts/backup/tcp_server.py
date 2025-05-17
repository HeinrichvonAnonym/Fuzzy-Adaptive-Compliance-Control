import rospy
import socket
import json
from geometry_msgs.msg import PoseArray, Pose
SERVER_IP = "192.168.31.127"
SERVER_PORT = 8081

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((SERVER_IP, SERVER_PORT))
sock.listen(1)

buffer = ""

rospy.init_node('tcp_server')
publisher = rospy.Publisher('/mrk/zed/human_frame', PoseArray, queue_size=10)
def publish_human_frame(landmarks):
    pa = PoseArray()
    pa.header.frame_id = "world"
    pa.header.stamp = rospy.Time.now()

    for lm in landmarks:
        pose = Pose()
        pose.position.x = lm['x']
        pose.position.y = lm['y']
        pose.position.z = lm['z']

        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1

        pa.poses.append(pose)
    
    publisher.publish(pa)


print("Server is listening...")
while (not rospy.is_shutdown()):
    conn, addr = sock.accept()
    print(f"Connected by {addr}")

    while (not rospy.is_shutdown()):
        data = conn.recv(1024)
        if not data:
            break
        # print(f"Received: {data.decode()}")
        buffer += data.decode()
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            try:
                landmarks = json.loads(line)
                publish_human_frame(landmarks)
            except json.JSONDecodeError:
                print("Invalid JSON format")

    conn.close()