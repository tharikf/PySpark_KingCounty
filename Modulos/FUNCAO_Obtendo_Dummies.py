
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType

def obtendo_dummies(spark_dataframe, coluna):
    
    # Obtendo as categorias de uma coluna
    categorias = spark_dataframe.select(coluna).distinct().rdd.flatMap(lambda x : x).collect()
    categorias.sort()

    # Obtendo as variáveis dummies
    df_alterado = spark_dataframe.select('*')

    for categoria in categorias:
        funcao = udf(lambda item: 1 if item == categoria else 0, IntegerType())
        nova_coluna = coluna + '_' + str(categoria).replace('.', '_')
        df_alterado = df_alterado.withColumn(nova_coluna, funcao(col(coluna)))
        
    # Deletando coluna utilizada
    df_alterado = df_alterado.drop(coluna)
    
    return df_alterado

