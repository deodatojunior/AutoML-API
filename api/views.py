import h2o
import pandas as pd

from rest_framework import generics
from rest_framework.response import Response

from .models import ModeloMachineLearningProcessado, ProcessamentoModeloMachineLearning
from .serializers import ProcessamentoModeloMachineLearningCreateSerializer, PrevisaoSerializer
# Create your views here.
class ProcessamentoModeloMachineLearningView(generics.CreateAPIView):
    serializer_class = ProcessamentoModeloMachineLearningCreateSerializer

    def perform_create(self, serializer):
        q = serializer.save()
        q.processar()

class PrevisaoView(generics.views.APIView):
    serializer_class = PrevisaoSerializer

    def post(self, request):
        try:
            model_id = request.POST.get('model_id')
            csv_prever = request.FILES['csv_prever']

            if model_id:
                modelo_processado = ModeloMachineLearningProcessado.objects.get(model_id=model_id)
                processamento = modelo_processado.processamentomodelomachinelearning_set.first()
            else:
                processamento = ProcessamentoModeloMachineLearning.objects.all().first()
                modelo_processado = processamento.modelos_processados.all().first()

            teste = pd.read_csv(csv_prever, sep=";")
            colunas_enviadas = ','.join(teste.columns.tolist())
            if processamento.variaveis_independentes != colunas_enviadas:
                raise Exception('Erro no layout do arquivo de previsão; Para este modelo são as seguintes colunas: "{variaveis_independentes}", mas você enviou as colunas: "{colunas_enviadas}"'.format(variaveis_independentes=processamento.variaveis_independentes, colunas_enviadas=colunas_enviadas))

            h2o.init()
            teste = h2o.H2OFrame(teste)

            modelo_automl = h2o.load_model(modelo_processado.binario_modelo.name)
            prever = modelo_automl.predict(teste)
            data_frame = prever.as_data_frame()

            previsoes = list()
            for i in range(0, len(data_frame['predict'])-1):
                previsoes.append({
                    'predict': data_frame['predict'][i],
                    'p0': data_frame['p0'][i],
                    'p1': data_frame['p1'][i]
                })

            return Response(status=201, data={'previsoes': previsoes})
        except Exception as e:
            return Response(status=401, data={'Erro': str(e)})

