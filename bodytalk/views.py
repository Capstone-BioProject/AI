from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db import connection
import numpy as np
import pandas as pd
import sys
from os import path
import json

# Message Recommend
@api_view(['POST', 'GET'])
def disease_recomm(request):
    print("views.py에서 실행")
    if request.method == 'POST':
        print("Django Success!")
        content = request.data.get('content') # Spring 요청 데이터
        sex=request.data.get('sex') 
        print("request data : " + content)
        print("request sex : ",sex)

        # KoBert 감정 분석 모델
        # model_result = [21.45123, 10.1234, 4.012312, 4.01234, 31.43234, 13.123415]
        #__file__는 현재 수행중인 코드를 담고 있는 파일의 위치를 담는다.
        sys.path.append(path.join(path.dirname(__file__), '..'))
        from kobert_predict import predict
        model_result = predict(content,sex)

        return Response(model_result, status=status.HTTP_200_OK)