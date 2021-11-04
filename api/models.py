from django.db import models

# Create your models here.

from decimal import Decimal

from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

import h2o
from h2o.automl import H2OAutoML

import pandas as pd

class ModeloMachineLearningProcessado(models.Model):
    model_id = models.TextField('Identificador do modelo', null=True)
    auc = models.DecimalField('aux', max_digits=10, decimal_places=6, null=True)
    logloss = models.DecimalField('logloss', max_digits=10, decimal_places=6, null=True)
    aucpr = models.DecimalField('aucpr', max_digits=10, decimal_places=6, null=True)
    mean_per_class_error = models.DecimalField('mean_per_class_error', max_digits=10, decimal_places=6, null=True)
    rmse = models.DecimalField('rmse', max_digits=10, decimal_places=6, null=True)
    mse = models.DecimalField('mse', max_digits=10, decimal_places=6, null=True)
    binario_modelo = models.FileField('Binário do modelo ML', upload_to='binario_modelo_ml', null=True)


class ProcessamentoModeloMachineLearning(models.Model):
    data = models.DateTimeField('Data e hora do processamento', auto_now_add=True)
    dados_csv = models.FileField('Arquivo CSV', upload_to='arquivos_csv')
    classe = models.CharField('Classe', max_length=30)
    variaveis_independentes = models.TextField('Variáveis independentes', null=True)
    tempo_maximo = models.PositiveIntegerField('Tempo máximo em segundos',
                                               validators=[MinValueValidator(settings.TEMPO_MINIMO_PROCESSAMENTO),
                                                           MaxValueValidator(settings.TEMPO_MAXIMO_PROCESSAMENTO)],)
    modelos_processados = models.ManyToManyField('api.ModeloMachineLearningProcessado')

    class Meta:
        ordering = ['-data']

    def processar(self):
        h2o.init()
        imp = pd.read_csv(self.dados_csv, sep=';')
        colunas = imp.columns.tolist()
        variaveis_independentes = [coluna for coluna in colunas if coluna != self.classe]
        self.variaveis_independentes = ','.join(variaveis_independentes)
        self.save()

        imp = h2o.H2OFrame(imp)
        treino, teste = imp.split_frame(ratios=[.7])

        treino[self.classe] = treino[self.classe].asfactor()
        teste[self.classe] = teste[self.classe].asfactor()

        modelo_automl = H2OAutoML(max_runtime_secs=self.tempo_maximo, sort_metric='AUC')
        modelo_automl.train(y=self.classe, training_frame=treino)

        ranking = modelo_automl.leaderboard
        ranking = ranking.as_data_frame()

        for i in range(8, len(ranking)-1):
            modelo_processado = ModeloMachineLearningProcessado()
            modelo_processado.model_id = ranking['model_id'].iloc[i]
            modelo_processado.auc = ranking['auc'].iloc[i].astype(Decimal)
            modelo_processado.logloss = ranking['logloss'].iloc[i].astype(Decimal)
            modelo_processado.aucpr = ranking['aucpr'].iloc[i].astype(Decimal)
            modelo_processado.mean_per_class_error = ranking['mean_per_class_error'].iloc[i].astype(Decimal)
            modelo_processado.rmse = ranking['rmse'].iloc[i].astype(Decimal)
            modelo_processado.mse = ranking['mse'].iloc[i].astype(Decimal)
            modelo = h2o.get_model(modelo_processado.model_id)
            modelo_processado.binario_modelo.name = h2o.save_model(modelo, path="%s/modelo" % settings.MEDIA_ROOT)
            modelo_processado.save()
            self.modelos_processados.add(modelo_processado)