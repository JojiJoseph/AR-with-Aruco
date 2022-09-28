import cv2
import cv2.aruco as aruco
import numpy as np
import pyrender
import trimesh

cap = cv2.VideoCapture(-1) # Replace with your camera index
cap.set(3, 1280)
cap.set(4, 720)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

def get_rot_matrix(alpha, beta, gamma):
    R1 = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)],
        ])
    R2 = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
        ])
    R3 = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1],
        ])
    return R3 @ R2 @ R1

roll180 = get_rot_matrix(np.pi, 0, 0)

# Replace with your camera matrix here
camera_matrix = np.array([
    [1430, 0, 480],
    [0,1430,620],
    [0,0,1]
], dtype=float)

camera = pyrender.IntrinsicsCamera(1430, 1430, 480, 620, znear=0.05, zfar=100.0, name=None)

cv2.namedWindow("Frame")

is_paused = False

camera_pose = None

fuze_trimesh = trimesh.load('./fuze.obj')
fuze_trimesh.apply_scale(20)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
renderer = pyrender.OffscreenRenderer(1280, 720)

while True:
    if not is_paused:
        ret, frame_orig = cap.read()
    frame = frame_orig.copy()
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict,
        parameters=aruco_params)
    aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs,_ = aruco.estimatePoseSingleMarkers(corners, 1, camera_matrix, None)
    if corners:
        for rvec, tvec in zip(rvecs, tvecs):
            aruco.drawAxis(frame, camera_matrix, np.array([[0,0.,0,0]]), rvec, tvec, 1)
        for t, p, r in zip(tvecs, corners, rvecs):
            r = r[0]
            t = t[0]

            R,_ = cv2.Rodrigues(r)
            Rg = roll180 @ R
            tg = roll180 @ t

            camera_pose = np.concatenate([ Rg.T, -Rg.T @ tg.reshape((3,1))], axis=1)
            camera_pose = np.concatenate([camera_pose,[[0,0,0,1]]], axis=0)
            
            if camera_pose is not None:
                scene = pyrender.Scene(ambient_light=[0.9,0.9,0.9])
                scene.add(mesh, camera_pose)
                scene.add(camera, pose=camera_pose)
                light = pyrender.PointLight(intensity = 500)
                scene.add(light, pose=camera_pose)
                color, depth = renderer.render(scene)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                mask = cv2.inRange(color, np.array([250,250,250]), np.array([255,255,255]))
                cv2.copyTo(color, 255-mask, frame)
    if is_paused:
        cv2.putText(frame, "Paused",(20,20),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        is_paused = not is_paused


cap.release()
cv2.destroyAllWindows()
