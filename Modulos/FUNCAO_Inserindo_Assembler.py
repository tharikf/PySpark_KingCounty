

from pyspark.ml.feature import VectorAssembler

def vetor_assembler(dataframe):
    
    # Listando as colunas e excluindo a coluna target (preco)
    colunas = dataframe.columns
    colunas = [coluna for coluna in colunas if coluna != 'preco']
    
    # Transformando colunas em um vetor
    assembler = VectorAssembler(inputCols = colunas, outputCol = "features")
    
    # Aplicando a transformação
    dataframe = assembler.transform(dataframe)
    
    return dataframe

