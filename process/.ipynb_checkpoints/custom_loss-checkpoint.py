import math
from six.moves import xrange

#tabnet focal loss
def focal_loss(y_pred, y_true):

    gamma = 2
    margin = 0.2
    weight_pos = 5
    weight_neg = 1

    em = np.exp(margin)
    y_pred = y_pred[:,1]
    log_pos = -F.logsigmoid(y_pred)
    log_neg = -F.logsigmoid(-y_pred)

    log_prob = y_true *log_pos + (1-y_true)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em+(1-em)*prob)
    weight = y_true * weight_pos + (1-y_true) * weight_neg
    loss = margin + weight * (1-prob)**gamma*log_prob

    return loss.mean()





class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        #approxes contains current predictions for this subset
        #targets contains target value you provided with the dataset
        #this function should return a list of pairs (der1, der2), where
        #der1 is the first derivative of the loss function with respect
        #to rhe predicted value, and der2 is the second derivative

        # Hyper Param : gamma
        
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)
        
        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = targets[index] - p
            der2 = -p * (1 - p)
            
            if (targets[index] == 0.0):
                # 타겟 값이 정상 유저라면, p값이 낮을수록 (정답에 가까울수록) 페널티가 작도록 세팅.
                der1 = (p+1)*(targets[index] - p)
                der2 = (p+1)*(-p * (1 - p))

            elif (targets[index] > 0.0):
                # 타겟 값이 fraud 이라면 p값이 높을수록 (정답에 가까울수록) 페널티가 적도록 세팅. 추가로 10% 더 페널티
                der1 = 4.5*(2-p)*(targets[index] - p) # p가 클수록 더 작은 페널티
                der2 = 4.5*(2-p)*(-p * (1 - p))
                
            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result