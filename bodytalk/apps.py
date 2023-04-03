from django.apps import AppConfig
import torch

class KobertmodelConfig(AppConfig):
    name = 'bodytalk'
    # GPU 사용
    device = torch.device("cpu")
    print(device)
    # 모델 load
    # PATH = '/Users/youn/SSAFY 문서/damhwa/kobert_model/'
    PATH = '/Users/정지원/Downloads/' # 배포용 (Docker에서 디렉터리 생성하기)
    model = torch.load(PATH + 'bertKoBERT_질병.pt', map_location=device)  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    model.load_state_dict(
        torch.load(PATH + 'bertmodel_state_dict.pt', map_location=device))  # state_dict를 불러 온 후, 모델에 저장
    print("————————model-load 완료————————")