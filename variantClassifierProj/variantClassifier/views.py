from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd

def home(request):
    return render(request, "home.html")

def result(request):
    # load model
    cls = joblib.load('final_model.sav')
    keep = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP', 'CLNVC','SIFT',
       'Allele', 'Consequence', 'IMPACT','CADD_PHRED', 'CADD_RAW']
    df = pd.DataFrame(columns=keep)
    df = df.append({'CHROM': pd.to_numeric(request.GET['CHROM']), 'POS': pd.to_numeric(request.GET['POS']),
     'REF': request.GET['REF'], 'ALT': request.GET['ALT'], 'AF_ESP': pd.to_numeric(request.GET['AF_ESP']),
      'AF_EXAC': pd.to_numeric(request.GET['AF_EXAC']), 'AF_TGP': pd.to_numeric(request.GET['AF_TGP']),
      'CLNVC': request.GET['CLNVC'],
      'SIFT': request.GET['SIFT'], 'Allele': request.GET['Allele'],
                'Consequence': request.GET['Consequence'], 'IMPACT': request.GET['IMPACT'],
                'CADD_PHRED': pd.to_numeric(request.GET['CADD_PHRED']),
                'CADD_RAW': pd.to_numeric(request.GET['CADD_RAW'])}, ignore_index=True)


    ans = cls.predict(df)
    return render(request, "result.html", {'ans': ans})
