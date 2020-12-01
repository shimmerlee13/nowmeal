from flask import Flask, request, render_template
from NowMeal import common as cm
from werkzeug.utils import secure_filename
import cx_Oracle
from flask import Flask,request,jsonify

import tensorflow as tf
import PIL.Image as pilimg
# ------recommenders system --------
import pandas as pd
import numpy as np



app = Flask(__name__)

# 레시피 상세화면
@app.route('/recipe_info', methods=['GET'])
def recipe_info():
    # key = request.form['key']
    recipe_name = request.args.get('key')
    other_recipe_image = request.args.get('key2')

    print(recipe_name)
    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    sql = "select cooking.pinfo from recipeinfo, cooking \
            where recipeinfo.rcode = cooking.rcode and recipeinfo.rname = :recipe_name"
    cursor.execute(sql, {'recipe_name' : recipe_name})
    cooking_info = cursor.fetchall()
    # {'value1': a[0], 'value2': a[1]}
    cursor.close()
    con.commit()
    con.close()
    kkk = []
    for cook in cooking_info:
        k = list(cook)
        kkk.append(k)

    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    sql = "select tagdetail.tdname from recipeinfo, reciperel, tagdetail \
    where tagdetail.tdid = reciperel.tdid \
    and reciperel.rcode = recipeinfo.rcode \
    and recipeinfo.rname = :recipe_name"
    cursor.execute(sql, {'recipe_name': recipe_name})
    tdname = cursor.fetchall()
    # {'value1': a[0], 'value2': a[1]}
    cursor.close()
    con.commit()
    con.close()
    kkkk = []
    for tag in tdname:
        k = list(tag)
        kkkk.append(k)
    return render_template('blog-single.html', name='blog_single', sibal=kkk, recipe_name=recipe_name, tdname=kkkk)



# 사용자 선택화면
@app.route('/user_info')
def user_select():
    return render_template('blog.html', name='blog')



# 메인페이지
@app.route('/')
def home():
    # dic = {"f1":list , "f2":list2}

    MYRESULT = [['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee']]
    return render_template('index.html', name = 'index', MYRESULT=MYRESULT)




# user 정보받는 페이지
@app.route('/uu1', methods = ['GET'])
def user():
    user_name = request.args.get('key')

    MYRESULT = [['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee']]
    mylist = cm.mylist()
    return render_template('index.html', name = 'index', MYRESULT = MYRESULT, user=user_name, MYLIST = mylist)






# 글씨 인식s
@app.route('/ss', methods = ['GET'])
def inputingre():

    receipt_result = cm.detect_text('C:/AIP/python-workspace/venv/NowMeal/image44.jpg')
    # print(receipt_result)
    # print(receipt_result[0][0])
    user_name = request.args.get('key')
    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    # print(type(receipt_result))
    sql = "insert into useringre(userid, inname) values("+ "'"+user_name + "'" +"," +"'"+receipt_result+ "'"+")"
    cursor.execute(sql)
    # {'value1': a[0], 'value2': a[1]}
    cursor.close()
    con.commit()
    con.close()

    MYRESULT = [['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee']]
    return render_template('index.html', name = 'index', MYRESULT = MYRESULT)



# webcam 연결 식재료 인식
@app.route('/dd', methods = ['GET'])
def our_cam():
    user_name = request.args.get('key')
    ingredient = cm.open_webcam()
    # print(ingredient)

    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    sql = "insert into useringre(userid, inname) values(" + "'" + user_name + "'" + "," + "'" + ingredient + "'" + ")"
    cursor.execute(sql)
    # {'value1': a[0], 'value2': a[1]}
    cursor.close()
    con.commit()
    con.close()

    MYRESULT = [['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee']]

    return render_template('index.html', name = 'index', MYRESULT = MYRESULT)


# 추천
@app.route('/tt', methods = ['GET'])
def recommend_start():
    user_name = request.args.get('key')
    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    sql = "select distinct inname from useringre where userid =:user_name "
    # print(str['INNAME'])
    cursor.execute(sql, {'user_name':user_name})
    user_ingre = cursor.fetchall()
    # print('real', list)
    u_i = []
    for i in user_ingre:
        for j in i:
            u_i.append(j)

    cursor.close()
    con.commit()
    con.close()
    user1_recomm = cm.recommend(user_name, u_i)
    list2 = []
    for i in range(len(user1_recomm)):
        kkk1 = []
        kkk1.append('assets/img/recipe_image1/' + user1_recomm.iloc[i, 0] + '.jpg.jpg')
        kkk1.append(user1_recomm.iloc[i, 0])
        kkk1.append('https://www.youtube.com/results?search_query='+ user1_recomm.iloc[i, 0])
        list2.append(kkk1)
    return render_template('index.html', name='index', MYRESULT=list2)




# text로 입력받기
@app.route('/kk', methods=['POST','GET'])
def insert_db():
    ingredient = request.form['ingredient']
    # print(ingredient)

    user_name = request.args.get('key')
    con = cx_Oracle.connect('recipe/0000@localhost:1521/xe')
    cursor = con.cursor()
    sql = "insert into useringre(userid, inname) values(" + "'" + user_name + "'" + "," + "'" + ingredient + "'" + ")"
    cursor.execute(sql)
    cursor.close()
    con.commit()
    con.close()
    MYRESULT = [['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee'],
                ['assets/img/white_back.png', 'recommend', 'lee']]
    return render_template('index.html', name = 'index', MYRESULT = MYRESULT)


if __name__ == '__main__':
    app.debug = True
    app.run()
#set FLASK_APP=pybo



