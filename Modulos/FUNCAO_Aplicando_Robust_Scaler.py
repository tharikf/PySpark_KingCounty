
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, RobustScaler
from pyspark.ml.functions import vector_to_array

def aplicando_robust_scaler(dataframe, lista_colunas):
    
    # Criando o VectorAssembler para transformar as colunas em vetores
    assembler = VectorAssembler(inputCols = lista_colunas, outputCol = "features")
    
    # Criando o RobustScaler para aplicar a escala robusta nas features
    scaler = RobustScaler(inputCol = "features", outputCol = "scaled_features", withScaling = True, withCentering = True)
    
    # Definindo a pipeline com as etapas de pré-processamento
    pipeline = Pipeline(stages = [assembler, scaler])
    
    # Treinando a pipeline com os dados de treinamento
    pipeline_model = pipeline.fit(dataframe)
    
    # Aplicando a pipeline no DataFrame de entrada
    dataframe = pipeline_model.transform(dataframe)
    
    # Dropando as colunas originais
    for coluna in lista_colunas:
        dataframe = dataframe.drop(coluna)
    
    # Dropando colunas originais no vetor
    dataframe = dataframe.drop('features')

    # Retornando para colunas
    dataframe = dataframe.withColumn('scaled',
                                     vector_to_array("scaled_features")) \
                         .select(dataframe.columns + [col('scaled')[i] for i in range(len(lista_colunas))])
    
    # Renomeando as colunas para o nome original
    colunas_retornar = dataframe.columns[-9:]
    for nome_original, nome_scaled in zip(lista_colunas, colunas_retornar):
        dataframe = dataframe.withColumnRenamed(nome_scaled, nome_original)
        
    # Dropando vetor scaled
    dataframe = dataframe.drop('scaled_features')
    
    # Dataframe final
    return dataframe