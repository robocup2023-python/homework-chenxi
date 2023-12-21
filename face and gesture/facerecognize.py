import os
import cv2
import insightface
import numpy as np
from sklearn import preprocessing
import mediapipe as mp
myHands= mp.solutions.hands
hands= myHands.Hands()
mpDraw = mp.solutions.drawing_utils
class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })

    # 人脸识别
    def recognition(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            for com_face in self.faces_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_name = com_face["user_name"]
            results.append(user_name)
        return results

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

    def register(self, image, user_name):
        faces = self.model.get(image)
        if len(faces) != 1:
            return '图片检测不到人脸'
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        is_exits = False
        for com_face in self.faces_embedding:
            r = self.feature_compare(embedding, com_face["feature"], self.threshold)
            if r:
                is_exits = True
        if is_exits:
            return '该用户已存在'
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
        self.faces_embedding.append({
            "user_name": user_name,
            "feature": embedding
        })
        return "success"

    # 检测人脸
    def detect(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()
            result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()
            result["pose"] = np.array(face.pose).astype(np.int32).tolist()
            result["age"] = face.age
            gender = '男'
            if face.gender == 0:
                gender = '女'
            result["gender"] = gender
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results


if __name__ == '__main__':
    image = cv2.imdecode(np.fromfile('test.jpg', dtype=np.uint8), -1)
    face_recognitio = FaceRecognition()
    # 人脸注册
    #result = face_recognitio.register(image, user_name='陈曦')
    #print(result)
    # 人脸识别
    results1 = face_recognitio.recognition(image)
    flag=[]
    for result in results1:
        print('识别结果：{}'.format(result))
        flag.append(format(result))
    results=face_recognitio.detect(image)
    num=0
    x=[]
    for result in results:
        a = format(result["bbox"])
        b = []
        c = 0
        for i in a:
            if i.isdigit():
                c = c * 10
                c = c + int(i)
            elif i == ',' or i == ']':
                b.append(c)
                c = 0
        print(b)
        x.append(b[0])
        x.append(b[1])
        image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0,0,255), 0)
        cv2.putText(image, format(flag[num]), (b[0],b[1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        num+=1
    mpDraw = mp.solutions.drawing_utils
    img_R = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_R)
    num=0
    # result.multi_hand_landmarks是检测到所有手的列表，对该列表进行访问我们可以得到每只手对应标志位的信息
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            gesture='no'
            mpDraw.draw_landmarks(image, handLms, myHands.HAND_CONNECTIONS)
            tip1 = handLms.landmark[4]
            tip2 = handLms.landmark[8]
            tip3 = handLms.landmark[12]
            tip4 = handLms.landmark[16]
            tip5 = handLms.landmark[20]
            dis12=((tip1.x-tip2.x)**2+(tip1.y-tip2.y)**2)**0.5
            dis23 = ((tip2.x - tip3.x) ** 2 + (tip2.y - tip3.y) ** 2) ** 0.5
            dis34 = ((tip3.x - tip4.x) ** 2 + (tip3.y - tip4.y) ** 2) ** 0.5
            dis45 = ((tip4.x - tip5.x) ** 2 + (tip4.y - tip5.y) ** 2) ** 0.5
            #print(dis12,dis23,dis34,dis45)
            if dis12<0.05:
                gesture='ok'
            elif dis23<0.1 and dis34>0.1:
                gesture='yes'
            cv2.putText(image, gesture, (x[num]+150,x[num+1]), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
            num+=2
    cv2.imwrite('result.jpg',image)