# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:09:01 2019

@author: Curt
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
from Bio import Entrez
import string
#import nltk
from collections import Counter
from plotly.offline import plot
import plotly.graph_objects as go
import flask

#df = pd.read_csv('C:/Users/Curt/Box Sync/Data Science/PubMed Project/Output/selfcontrol_test.txt')

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pasthname = '/app', external_stylesheets=external_stylesheets)

def get_keywords(xml_data):
        kewWordList = []
        for article in xml_data:
            tempKewWordList = []
            try:
                kewWords = article['MedlineCitation']['KeywordList'][0]
                for keyword in kewWords:
                    tempKewWordList.append(str(keyword)) #The str part is critical here; otherwise, you're still left with the strange Entrez-specific object type.
            except:
                tempKewWordList.append('empty')
                print('No keywords for this article')

            kewWordList.append(tempKewWordList) 
        return kewWordList
    
def get_dois(xml_data, article_idtype='doi'):
        doiList = []
        for i, article in enumerate(xml_data):
            try:
                dois = article['PubmedData']['ArticleIdList']
                for doi in dois:
                    if doi.attributes['IdType'].lower() == article_idtype:
                        doiList.append(str(doi))
            except: 
                doiList.append(str('empty'))
            if len(doiList) != i+1:
                doiList.append(str('empty'))
        return doiList  
        
def get_abstracts(xml_data):
    abstractsList = []
    for article in xml_data:
        try:
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            abstractsList.append(str(abstract))
        except:
            abstractsList.append('empty')
            print('No abstract for this article')
        #else:
         #   abstractsList.append('empty')
    return abstractsList      

def get_titles(xml_data):
    titlesList = []
    for article in xml_data:
        titles = article['MedlineCitation']['Article']['ArticleTitle']
        titlesList.append(str(titles))
    return titlesList       

def get_years(xml_data):
    yearsList = []
    for article in xml_data:
        try:
            year = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year']
            yearsList.append(str(year))
        except:
            yearsList.append('empty')
            print('No year for this article')
    return yearsList   

def get_journals(xml_data):
    journalsList = []
    for article in xml_data:
        journals = article['MedlineCitation']['Article']['Journal']['Title']
        journalsList.append(str(journals))
    return journalsList   

def get_authors(xml_data):
    authorList = []
    for article in xml_data:
        try:
            tempAuthorList = article['MedlineCitation']['Article']['AuthorList']
            tempNameList = []
            for i in range(len(tempAuthorList)):
                try:
                    last = tempAuthorList[i]['LastName']
                    first = tempAuthorList[i]['ForeName']
                    tempNameList.append(str(last) + ', ' + str(first))
                except:
                    print('No author for this author-entry')
            authorList.append(tempNameList)
        except:
            print('No authorlist for this article')
            authorList.append(str('empty'))
    return authorList 

def get_pages(xml_data):
    pagesList = []
    for article in xml_data:
        try:
            pages = article['MedlineCitation']['Article']['Pagination']['MedlinePgn']
            pagesList.append(str(pages))
        except:
            pagesList.append('empty')
            print('No pages for this article')
        #else:
        #    pagesList.append('empty')
    return pagesList 

#stopwords = nltk.corpus.stopwords.words('english')
stopwords = ['empty','association', 'associated', 'study', 'studies', 'task', 'may', 'effect', 
               'results', 'high', 'higher', 'low', 'lower', 'participants', 'findings',
               'related', 'however', 'findings', '=', 'also', 'levels', 'research', 'use',
               'whether', 'effects', 'differences', 'among', 'using', 'two', 'important',
               'evidence', 'found', 'tasks', 'relationship', 'examined', 'across',
               'negative', 'predicted', 'behavior', 'behaviors', 'associations',
               'group', 'test', 'p', 'examine', 'behavioral', 'significant', 'compared',
               'skills', 'significantly', 'greater', 'suggest', 'suggests', 'less', 'reduced',
               'one', 'would', 'many', 'r', 'three', 'two', 'large', 'paper', 'e.g.',
               'showed', '1', '2']
#stopwords.extend(myStopWords)

colors = ['#FFCDB2', '#E8A494', '#E5989B', '#B5838D', '#6D6875']

# =============================================================================
# =============================================================================
# =============================================================================
app.layout = html.Div([
    html.Div([
        dcc.Input(id='input', value='self-control', type='text'),
        html.Button(id='submit-button', n_clicks=0, children='Submit')],
    style={'textAlign': 'center'}),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='graph_abstract'),
                dcc.Graph(id='graph_keywords')],
            style={'margin': 'auto', 'display': 'inline-block', 'width': '100%'})]),
        html.Div(dcc.Graph(id='graph_scatter', clickData={'points': [{'customdata': '2018'}]}), 
        style={'margin': 'auto', 'display': 'inline-block', 'width': '100%'})],
    style={'columnCount': 2, 'columnGap': '0%', 'columnRule': '4px'}),
    html.Div([
        dcc.Graph(id='graph_journal'),
        dcc.Graph(id='graph_author')],        
    style={'columnCount': 2, 'columnGap': '0%'}),
    html.Div(id='table')
    ])
# =============================================================================
# =============================================================================
# =============================================================================
    
@app.callback(Output('intermediate-value', 'children'), 
              [Input(component_id='submit-button', component_property='n_clicks')],
              [State(component_id='input', component_property='value')])
def get_data(n_clicks, query):
    query = str(query)
    howmany = 500
    #sort = 'relevance'
    #field = None
    searchResults = Entrez.read(Entrez.esearch(db='pubmed', retmode='xml', term=query, retmax=howmany, sort='relevance'))                                        
    ids = str(searchResults['IdList'])
    handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
    xml_data = Entrez.read(handle)['PubmedArticle']
    titles = get_titles(xml_data)
    keywords = get_keywords(xml_data)
    abstracts = get_abstracts(xml_data)
    years = get_years(xml_data)
    journals = get_journals(xml_data)
    authors = get_authors(xml_data)
    pages = get_pages(xml_data)
    dois = get_dois(xml_data)
    
    df = pd.DataFrame()
    df['titles'] = titles
    df['keywords'] = keywords
    df['abstracts'] = abstracts
    df['years'] = years
    df['journals'] = journals
    df['authors'] = authors
    df['pages'] = pages
    df['dois'] = dois
    df['query'] = query
    df['number_extracted'] = len(df)
    
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols] 
    return df.to_json(orient='split')

@app.callback(Output('table', 'children'), [Input('intermediate-value', 'children')])
def update_table(jsonified_cleaned_data, max_rows=10):
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    dataframe = dataframe[['titles','keywords','years','journals','authors']]
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

#@app.callback([Output('year-slider', 'min'),
#               Output('year-slider', 'max'),
#               Output('year-slider', 'value'),
#               Output('year-slider', 'marks'),
#               Output('year-slider', 'step')],
#              [Input('intermediate-value', 'children')])
#def set_slider(jsonified_cleaned_data):
#    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
#    df = dataframe[['years']]
#    df = df[df['years'] != "empty"]
#    min=df['years'].min()
#    max=df['years'].max()
#    value=df['years'].max()
#    marks={str(year): str(year) for year in df['years'].unique()}
#    step=None
#    return int(min), int(max), int(value), marks, step
        
@app.callback(Output('graph_abstract', 'figure'), 
              [Input('intermediate-value', 'children'),
               Input('graph_scatter', 'clickData')])
def update_abstract_graph(jsonified_cleaned_data, clickData):   
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    filtered_df = dataframe[dataframe['years'] == clickData['points'][0]['customdata']]
    abs_list = filtered_df['abstracts']
    abstracts = []
    for abstract in abs_list:
        temp = ''.join(ch for ch in abstract if ch not in string.punctuation)
        abstracts.append(temp)
    abstracts_joined = " ".join(abstract for abstract in abstracts)
    abstracts_tokens = [t.lower() for t in abstracts_joined.split()]
    clean_abstracts_tokens = abstracts_tokens[:]
    for token in abstracts_tokens:
        if token in stopwords:
            clean_abstracts_tokens.remove(token)
    word_counts = Counter(clean_abstracts_tokens)       
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    freq_df.sort_values(by=['count'], inplace=True, ascending=False)
    #freq_df = freq_df[freq_df['count'] >= 10]
    freq_df = freq_df.iloc[:30]
    freq_df = freq_df[freq_df['count'] != max(freq_df['count'])]
    rem = len(freq_df) % 5
    freq_df_rem = freq_df.iloc[:-rem]
    NinBin = int(len(freq_df_rem) / 5)
    binIndices = [int(x * NinBin) for x in range(1,6)]
    freq_df_binned = [freq_df_rem.iloc[x-NinBin:x] for x in binIndices]
    return {
        'data': [go.Bar(
            x = list(df['word'].values),
            y = list(df['count'].values),
            marker={'color': colors[i]})
            for i, df in enumerate(freq_df_binned)],
        'layout': {
            'yaxis': {'title': 'Word frequency'},
            'height': 275,
            'margin': {'l': 40, 'b': 85, 'r': 0, 't': 40},
            'title': 'Abstracts: word frequency',
            'showlegend': False,
            #'annotations': [{
            #'x': 0.80, 'y': 0.75, 'xanchor': 'left', 'yanchor': 'bottom',
            #'xref': 'paper', 'yref': 'paper', 'showarrow': False,
            #'align': 'right', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
            #'text': '2018'}]
            }
        }

@app.callback(Output('graph_keywords', 'figure'), 
              [Input('intermediate-value', 'children'),
               Input('graph_scatter', 'clickData')])
def update_keyword_graph(jsonified_cleaned_data, clickData):   
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    filtered_df = dataframe[dataframe['years'] == clickData['points'][0]['customdata']]
    key_ser = filtered_df['keywords']
    key_strings = [','.join(words) for words in key_ser]
    key_joined = ','.join(words for words in key_strings)
    keywords_tokens = [t.lower() for t in key_joined.split(',')]
    word_counts = Counter(keywords_tokens)       
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    freq_df = freq_df[freq_df['word'] != 'empty']
    freq_df.sort_values(by=['count'], inplace=True, ascending=False)
    #freq_df = freq_df[freq_df['count'] >= 10]
    freq_df = freq_df.iloc[:12]
    freq_df = freq_df[freq_df['count'] != max(freq_df['count'])]
    rem = len(freq_df) % 5
    freq_df_rem = freq_df.iloc[:-rem]
    NinBin = int(len(freq_df_rem) / 5)
    binIndices = [int(x * NinBin) for x in range(1,6)]
    freq_df_binned = [freq_df_rem.iloc[x-NinBin:x] for x in binIndices]
    return {
        'data': [go.Bar(
            x = list(df['word'].values),
            y = list(df['count'].values),
            marker={'color': colors[i]})
            for i, df in enumerate(freq_df_binned)],
        'layout': {
            'yaxis': {'title': 'Word frequency'},
            'height': 275,
            'margin': {'l': 40, 'b': 85, 'r': 0, 't': 40},
            'title': 'Keywords: word frequency',
            'showlegend': False,
            #'annotations': [{
            #'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
            #'xref': 'paper', 'yref': 'paper', 'showarrow': False,
            #'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
            #'text': '2018'}]
            }
        }

@app.callback(Output('graph_scatter', 'figure'), 
              [Input('intermediate-value', 'children')])
def graph_scatter(jsonified_cleaned_data):    
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    filtered_df = dataframe[dataframe['years'] != 'empty']
    count_df = filtered_df[['titles','years']].groupby('years', as_index=False).count()
    return {
        'data': [go.Scatter(
            x = count_df['years'],
            y = count_df['titles'],
            mode='lines+markers',
            marker={'color': '#6D6875'},
            customdata = count_df['years'])],
        'layout': {
            'height': 375,
            'padding': 0,
            'title': 'Number of publications by year',
            'showlegend': False,
            'margin': {'l': 30, 'b': 15, 't': 100, 'r': 10},
            'hovermode': 'closest'
            }
        }

@app.callback(Output('graph_journal', 'figure'), 
              [Input('intermediate-value', 'children')])
def graph_journal(jsonified_cleaned_data):    
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    filtered_df = dataframe[dataframe['journals'] != 'empty']
    count_df = filtered_df[['titles','journals']].groupby('journals', as_index=False).count()
    count_df.sort_values(by=['titles'], inplace=True, ascending=True)
    count_df = count_df.iloc[-12:]
    return {
        'data': [go.Bar(
            x = count_df['titles'], 
            y = count_df['journals'],
            marker={'color': '#B5838D'},
            orientation='h',
            )],
        'layout': {
            'xaxis': {'title': 'Frequency'},
            'height': 350,
            'margin': {'l': 450, 'b': 40, 'r': 0, 't': 25, 'pad': 5},
            'title': 'Most Common Journals',
            'showlegend': False,
            }
        }
        
@app.callback(Output('graph_author', 'figure'), 
              [Input('intermediate-value', 'children')])
def graph_author(jsonified_cleaned_data):    
    dataframe = pd.read_json(jsonified_cleaned_data, orient='split')
    auth_ser = dataframe['authors']
    #auth_ser = auth_ser[auth_ser['authors'] != 'empty']
    auth_strings = ['@'.join(words) for words in auth_ser]
    auth_joined = '@'.join(words for words in auth_strings)
    auth_tokens = [t.lower() for t in auth_joined.split('@')]
    word_counts = Counter(auth_tokens)       
    freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
    freq_df = freq_df[freq_df['word'] != 'empty']
    freq_df.sort_values(by=['count'], inplace=True, ascending=True)
    #freq_df = freq_df[freq_df['count'] >= 10]
    freq_df = freq_df.iloc[-12:]
    return {
        'data': [go.Bar(
            x = freq_df['count'], 
            y = freq_df['word'],
            marker={'color': '#E8A494'},
            orientation='h',
            )],
        'layout': {
            'xaxis': {'title': 'Frequency'},
            'height': 350,
            #'margin': {'l': 250, 'b': 30, 'r': 0, 't': 25},
            'margin': go.layout.Margin(
                    l=200,
                    r=0,
                    b=40,
                    t=25,
                    pad=5),
            'title': 'Most Common Authors',
            'showlegend': False,
            }
        }

###other app setup code, i.e., the flask wrapper for the dash app###
@server.route('/')
def index():
    return '''
<html>
<div>
    <h1>Flask App</h1>
</div>
</html>
'''

if __name__ == '__main__':
    server.run(debug=True)