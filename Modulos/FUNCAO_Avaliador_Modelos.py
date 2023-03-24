
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator

def avaliador_modelos(modelo, dados_treino, dados_teste, nome_modelo):
    
    # Fit
    fit = modelo.fit(dados_treino)
    
    # Previsao
    previsao = fit.transform(dados_teste)
    
    # Avaliador
    rmse = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'preco', metricName = 'rmse').evaluate(previsao)
    r2 = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'preco', metricName = 'r2').evaluate(previsao)
    
    # Apresentando os resultados
    df = pd.DataFrame({'Algoritmo' : nome_modelo,
                       'RMSE' : ['{:.4f}'.format(rmse)],
                       'R2' : ['{:.4f}'.format(r2)]})
    
    return df
