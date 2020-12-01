# import sqlite3
import cx_Oracle
import pandas as pd
import numpy as np
# -----바코드
from google.cloud import vision
import io
import os
# -----recommenders
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from numpy import linspace
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.decomposition import PCA
import os

def mylist():
    conn = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cur = conn.cursor()
    cur.execute("select inname from useringre where userid = 'demo1'")
    rows = cur.fetchall()
    list = []
    for row in rows:
        list.append(row)
    conn.close()
    return list

# -------영수증 인식 API-------
def detect_text(IMG_PATH):
    credential_path = "springsaturday-d6504b76ac76.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    client = vision.ImageAnnotatorClient()

    with io.open(IMG_PATH, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    #     print('Texts:')

    txt = []
    for text in texts[1:]:
        #         print('\n"{}"'.format(text.description))
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])
        #         print('bounds: {}'.format(','.join(vertices)))
        txt.append(text.description)

    if response.error.message:
        raise Exception('error : {}'.format(response.error.message))

    return txt




# --------recommenders syspem ------------------
def recommend(user_id, ingred):
    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    tag = pd.read_sql('select reciperel.rcode, recipeinfo.rname, tagdetail.tdname \
                      from recipeinfo, reciperel, tagdetail \
                      where recipeinfo.rcode = reciperel.rcode and reciperel.tdid = tagdetail.tdid',
                      con=con).sort_values(by = 'RCODE')
    cursor.close()
    con.close()
    con = cx_Oracle.connect("recipe/0000@localhost:1521/xe")
    cursor = con.cursor()
    user_tag = """select userrel.userid, tagdetail.tdname from userrel, tagdetail \
                where userrel.tdid = tagdetail.tdid and userrel.userid = :userid"""
    cursor.execute(user_tag, {'userid':user_id})
    user_tag = pd.DataFrame(cursor.fetchall(), columns=['USERID', 'TDNAME'])
    cursor.close()
    con.close()

    user_tag_list = user_tag['TDNAME']

    tag_dummies = pd.get_dummies(tag['TDNAME'])
    tag = pd.concat([tag,tag_dummies], axis=1)
    tag = tag.drop(tag[['RCODE','TDNAME']], axis=1)
    tag = tag.groupby('RNAME').sum()

    # 재료이름, 레시피코드, 레시피이름 테이블 가져와서
    # print(ingred[0])

    con = cx_Oracle.connect("recipe/0000@localhost:1521/xe")
    cursor = con.cursor()
    inrel = "select recipeinfo.rname, inrel.INNAME  from recipeinfo, inrel \
                            where inrel.inname in" + "("+"'" +ingred[0] + "'" +"," + "'" + ingred[1]+ "'" +")" +" \
                            and inrel.RCODE = recipeinfo.RCODE"
    cursor.execute(inrel)
    ingredient = pd.DataFrame(cursor.fetchall(), columns=['RNAME', 'INNAME'])
    cursor.close()
    con.close()

    ingredient.index=ingredient['RNAME']
    # tag['RNAME'], ingredient['RNAME'] join
    tag = tag.loc[ingredient.index, :]

    kmeans = KMeans()
    pred = kmeans.fit_predict(tag)
    tag['pam_fit.clustering'] = pred
    user = pd.DataFrame(columns = tag.columns[0:65])
    for i in user_tag_list:
        user.loc[user_id, i] = 1
        user = user.fillna(0)
    pred_u = kmeans.predict(user)
    user_clus = tag[tag['pam_fit.clustering'] == pred_u[0]]
    user_clus_train = user_clus.drop(user_clus[['pam_fit.clustering']],axis=1)
    user_clus_train = pd.concat([user, user_clus_train])
    user_cos = cosine_similarity(user_clus_train, user_clus_train)
    user_cos_index = user_cos.argsort()[:, ::-1]
    recommend_ind = user_cos_index[:1]
    recommend_ind = recommend_ind[0][:11]
    recommend_top10 = pd.DataFrame(user_clus_train.iloc[recommend_ind, ].index)
    return recommend_top10[1:10]





# def need_ingred(reco_list):
#     reco = reco_list.iloc[0, 0]
#     con = cx_Oracle.connect("recipe/0000@localhost:1521/xe")
#     cursor = con.cursor()
#     need_in = """select inrel.rcode, recipeinfo.rname, inrel.inname, inrel.inrole from inrel, recipeinfo \
#                 where inrel.rcode = recipeinfo.rcode \
#                 and inrole = '주재료' \
#                 and recipeinfo.rname = :reco"""
#     cursor.execute(need_in, {'reco': reco})
#     need_in = pd.DataFrame(cursor.fetchall(), columns=['RCODE', 'RNAME', 'INNAME', 'INROLE'])
#     cursor.close()
#     con.close()
#     need_ingredient = need_in['INNAME']
#     return reco_list.iloc[0, 0], need_ingredient
#
# recommend_menu, need_ingredient = need_ingred(reco_list)
# need_ingredient




from keras.models import load_model
# inception_model = load_model('C:/AIP/python-workspace/venv/model-inception-33.hdf5')
print('inceptionV3 모델 연결 완료')

my_dict = {'carrot': 0, 'egg': 1, 'onion': 2, 'radish': 3, 'rpepper': 4}

new_dict = {}
for k, v in my_dict.items():
    new_dict[v] = k
print('-------환경세팅완료-----')

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import models
import math

# Load the saved model
def open_webcam():
    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()
        # Capture Frame을 RGB로 변환
        image = Image.fromarray(frame, 'RGB')
        image = np.array(image)  # (480, 640, 3)   center
        (height, width) = image.shape[:2]  # height = 480 , width = 640

        image_slicing = image[120:360, 160:480]  # image slicing (이미지 부분 추출)

        ################################### 파란색 직사각형그리기 #######################################################
        #         cv2.rectangle(frame,(160,120,width//2,height//2),(255,0,0),4)  # 파란색으로 직사각형 그리는 부분

        ############################### [ Frame 에서 객체 추출 및 객체 bounding box 작업 ] ##############################
        # ----------------------------------------------------------------------------------------------------------------------------
        dst = cv2.cvtColor(image_slicing, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        dst = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

        _, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
        # print(stats)

        count = 0
        obj_list = []
        for x, y, w, h, cnt in stats:
            frame_obj_x = x + 160
            frame_obj_y = y + 120
            frame_obj_center_x = frame_obj_x + (h // 2)
            frame_obj_center_y = frame_obj_y + (w // 2)
            distance_between_obj_frame_center = math.sqrt(
                ((frame_obj_center_x - 240) ** 2) + ((frame_obj_center_y - 320) ** 2))
            if (h, w) < (240, 320) and w < 320 and h < 240 and cnt > 1000 and distance_between_obj_frame_center < 100.0:
                kk = []
                kk.append(frame_obj_x)
                kk.append(frame_obj_y)
                kk.append(w)
                kk.append(h)
                obj_list.append(kk)
                #                 cv2.rectangle(frame,(frame_obj_x,frame_obj_y,w,h),(0,0,255),2) ## 객체인 부분만 직사각형 그림
                count += 1

        # -------------------------------------------------------------------------------------------------------------------------------
        ########################################## [ Predict ] ############################################################

        if len(obj_list) == 1:
            px, py, pw, ph = obj_list[0]
            #             image_slicing2 = image[px:px+ph, py:py+pw] # image slicing (이미지 부분 추출)
            #             image_slicing2 = cv2.resize(image_slicing2, dsize=(128, 128), interpolation=cv2.INTER_LINEAR) #         훈련한 이미지가 128x128이기 때문에 frame을 input shape과 동일하게 설정
            #             image_slicing2 = image_slicing2 / 255.0
            #             image_slicing2 = image_slicing2[np.newaxis, :]
            image_slicing = cv2.resize(image_slicing, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
            image_slicing = image_slicing / 255.0
            image_slicing = image_slicing[np.newaxis, :]
            predict = inception_model.predict(image_slicing)

            obj_name = new_dict[np.argmax(predict)]
            if obj_name == 'rpepper':
                cv2.putText(frame, obj_name, (px - 25, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (px, py, pw, ph), (0, 255, 0), 2)  ## 객체인 부분만 직사각형 그림

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return '홍고추'

# open_webcam()

# q exit
