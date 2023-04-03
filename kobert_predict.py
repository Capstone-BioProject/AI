from torch.utils.data import Dataset
from manage import *
from bodytalk.apps import KobertmodelConfig

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

disease=['ADHD', 'A형 간염', '각막염', '감기', '건선', '결핵', '고혈압', '골다공증', '골절', '공황장애', '궤양성대장염', '기흉',
 '녹내장', '뇌전증', '다태아 임신', '당뇨병', '류마티스 관절염', '만성폐쇄성폐질환', '목 디스크', '방광염', '백내장', '백혈병', '변비',
 '분만건수', '불면증', '불임', '비만', '비염', '빈혈', '성조숙증', '소화불량', '수족냉증', '식중독', '심근경색증','아토피 피부염', '안구건조증', 
 '알레르기성 결막염', '알코올중독증', '오십견', '요로결석', '요실금', '우울증', '인플루엔자', '자궁근종', '장염','접촉성 피부염', '조기진통', 
 '조울증', '중이염', '지방간', '질염', '충치', '치매', '치은염', '치질', '코로나', '탈모', '통풍', '패혈증', '협심증', '화상']

def new_softmax(a):
    c = np.max(a)  # 최댓값
    exp_a = np.exp(a - c)  # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


# 예측 모델 설정
# 예측 모델 설정
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    # 토큰화
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    # Setting parameters
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 5
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5
    print("————————parameter 세팅 완료————————")


    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    KobertmodelConfig.model.eval()

    for token_ids, valid_length, segment_ids, label in test_dataloader:
        token_ids = token_ids.long().to(KobertmodelConfig.device)
        segment_ids = segment_ids.long().to(KobertmodelConfig.device)

        valid_length= valid_length
        label = label.long().to(KobertmodelConfig.device)

        out = KobertmodelConfig.model(token_ids, valid_length, segment_ids)

        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
           # for logit in logits:
            #    print(logit, end=' ')
             #   probability.append(np.round(logit, 3))
            for i in range(0,61):
                if np.argmax(logits) == i: 
                  probability.append(disease[i])
            print(probability)

    return probability