from rest_framework import serializers

from .models import ProcessamentoModeloMachineLearning, ModeloMachineLearningProcessado

class ModeloMachineLearningProcessadoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModeloMachineLearningProcessado
        fields = ['model_id', 'auc', 'logloss', 'aucpr', 'mean_per_class_error', 'rmse', 'mse', 'binario_modelo']

class ProcessamentoModeloMachineLearningCreateSerializer(serializers.ModelSerializer):
    modelos_processados = ModeloMachineLearningProcessadoSerializer(many=True, read_only=True)

    class Meta:
        model = ProcessamentoModeloMachineLearning
        fields = ['id', 'data', 'dados_csv', 'classe', 'variaveis_independentes', 'tempo_maximo','modelos_processados']
        read_only_fields = ['id','data', 'variaveis_independentes', 'modelos_processados']


class PrevisaoSerializer(serializers.Serializer):
    model_id = serializers.CharField(required=False)
    csv_prever = serializers.FileField(allow_empty_file=False)