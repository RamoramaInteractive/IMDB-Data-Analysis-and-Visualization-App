#
#	IMDB Data Analysis and Visualization App
#
#	Author: Ramón Wilhelm
#	Date:	24/05/2021
#

import pandas as pd
import numpy as np
from numpy import random

import plotly.graph_objects as go

import networkx as nx
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_table

start = time.time()

genres_obj = []

data = pd.read_csv('imdb_top_1000.csv', skiprows=0, low_memory=False)


data.Poster_Link = data.Poster_Link.replace(to_replace = np.nan, value = "-")
data.Series_Title = data.Series_Title.replace(to_replace = np.nan, value = "-")
data.Released_Year = data.Released_Year.replace(to_replace = np.nan, value = -1)
data.Certificate = data.Certificate.replace(to_replace = np.nan, value = "-")
data.Runtime = data.Runtime.replace(to_replace = np.nan, value = "-")
data.Genre = data.Genre.replace(to_replace = np.nan, value = "-")
data.IMDB_Rating = data.IMDB_Rating.replace(to_replace = np.nan, value = -1)
data.Overview = data.Overview.replace(to_replace = np.nan, value = "-")
data.Meta_score = data.Meta_score.replace(to_replace = np.nan, value = -1)
data.Director = data.Director.replace(to_replace = np.nan, value = "-")
data.Star1 = data.Star1.replace(to_replace = np.nan, value = "-")
data.Star2 = data.Star2.replace(to_replace = np.nan, value = "-")
data.Star3 = data.Star3.replace(to_replace = np.nan, value = "-")
data.Star4 = data.Star4.replace(to_replace = np.nan, value = "-")
data.Gross = data.Gross.replace(to_replace = np.nan, value = -1)

removed_nodes = []
removed_movies = []

'''
Attributes for the graph
'''
iter = 80
distance_between_nodes = 0.2

#'''
#Get all the rows where the leading actors (Star1) have played in a certain number of movies
#and group them by the leading actors.
#'''
star_played_movies_count = 3

star1_count = data.Star1.value_counts()
star1_data = data[data["Star1"].isin(star1_count[star1_count >= star_played_movies_count].index)]
star1_data = star1_data.sort_values('Star1')
star1_group = star1_data.groupby('Star1')

'''
Define the graph for the nodes and edges.
'''
G = nx.Graph()

'''
Get from each leading star their movies and supportive actors (Star2)
and add all the stars in nodes. For the edges add the stars in nodes and a list of movies
where both of the stars appear.
'''

def add_nodes_and_edges_of_actors_to_the_graph(graph, star_group):
	graph.clear()
	for key, item in star_group: #star1_group:
		for star2 in item.Star2:
			movies = data[data['Star1'] == key]
			movies = movies[movies['Star2'] == star2]
			movies.sort_values('Series_Title', ascending=False)
			#movies = movies.Series_Title.to_numpy()
			movies = movies.loc[:, ['Series_Title', 'Released_Year', 'Genre', 'Runtime', 'Certificate', 'IMDB_Rating', 'Meta_score', 'Director', 'No_of_Votes', 'Gross', 'Star1', 'Star2']]
			for movie in movies.Series_Title:
				graph.add_node(key)
				graph.add_node(star2)
				graph.add_edge(key, star2, movies=movies)			
			
def remove_actors_by_centrality(graph, cent):
	copy_of_the_centralities = nx.degree_centrality(G)

	centralities = pd.DataFrame.from_dict({
		'Stars': list(copy_of_the_centralities.keys()),
		'Centrality': list(copy_of_the_centralities.values())
	})

	centralities = centralities.sort_values('Centrality', ascending=False)

	unnormalized_c = []

	for c in centralities.Centrality:
		unnormalized_c.append((int(c * centralities.shape[0])))

	centralities.Centrality = unnormalized_c
	
	for key, item in zip(centralities.Stars, centralities.Centrality):
		if item <= cent:
			graph.remove_node(key)
			
add_nodes_and_edges_of_actors_to_the_graph(G, star1_group)

'''
Generate positions by the layout and assign them to the nodes. 
'''		
pos = nx.spring_layout(G, k=distance_between_nodes, iterations=iter)
nx.set_node_attributes(G, pos, 'pos')

'''
Defining an edge trace for plotting the edges.
'''
edge_trace = go.Scatter(
	x=[],
	y=[],
	line=dict(width=1, color='#888'),
	hoverinfo='none',
	mode='lines')
	
'''
Defining an edge trace for plotting the edges.
'''	
for edge in G.edges():
	x0, y0 = G.nodes[edge[0]]['pos']
	x1, y1 = G.nodes[edge[1]]['pos']
	edge_trace['x'] += tuple([x0, x1, None])
	edge_trace['y'] += tuple([y0, y1, None])
	
'''
Defining node trace for plotting the nodes.
'''		
node_trace = go.Scatter(
	x=[],
	y=[],
	text=[],
	mode='markers',
	hoverinfo='text',
	marker=dict(
		showscale=True,
		colorscale='Oranges_r',
		reversescale=True,
		color=[],
		size=15,
		colorbar=dict(
			thickness=10,
			title='Centrality of the leading actors',
			xanchor='left',
			titleside='right'
		),
		line=dict(width=0)))

'''
Add node positions to the node trace.
'''				
for node in G.nodes():
	x, y = G.nodes[node]['pos']
	node_trace['x'] += tuple([x])
	node_trace['y'] += tuple([y])
			
'''
Add all information about each node to the node trace.
'''			
for node, adjacencies in enumerate(G.adjacency()):
	node_trace['marker']['color']+=tuple([len(adjacencies[1])])
	node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
	node_trace['text']+=tuple([node_info])
	
graph = go.Figure(data=[edge_trace, node_trace],
				 layout=go.Layout(
					title='',
					titlefont=dict(size=16),
					showlegend=False,
					hovermode='closest',
					margin=dict(b=20,l=5,r=5,t=40),
					annotations=[dict(
						text="",
						showarrow=False,
						xref="paper", yref="paper")
						],
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
))
graph.update_layout({'plot_bgcolor': 'rgb(137, 207, 237)'})		

orig_graph = G.copy()

'''
Calculating the centralities for each node by using the nx.egree_centrality() function

Note: The values of the nx.degree_centrality() function is normalized. 
In order to get the centralities as non-normalized values you have to muliply them by the number of all actors
'''

def get_descending_degree_centralities_of_stars():
	copy_of_the_centralities = nx.degree_centrality(G)

	centralities = pd.DataFrame.from_dict({
		'Stars': list(copy_of_the_centralities.keys()),
		'Centrality': list(copy_of_the_centralities.values())
	})

	centralities = centralities.sort_values('Centrality', ascending=False)

	unnormalized_c = []

	for c in centralities.Centrality:
		unnormalized_c.append((int(c * centralities.shape[0])))

	centralities.Centrality = unnormalized_c
	
	return centralities
	
def get_min_and_max_centrality(centralities):
	max = centralities.iloc[0]['Centrality']
	min = centralities.iloc[len(centralities.index)-1]['Centrality']
	return min, max

'''
Caclulatiing Histogram Degrees.
'''
def get_histogram_degrees(graph):
	degree_freq = nx.degree_histogram(graph)
	degrees = range(len(degree_freq))
	return degrees

'''
Betweenness Centrality (Number of shortest paths in a network that pass through a particular node)
'''
'''
print("Betweenness Centrality:")
D = nx.betweenness_centrality(G, k=311)
L = sorted(D.items(), key=lambda item: item[1], reverse=True)
for i in range(5):
    print(L[i][0], ":", L[i][1])
print()
'''



'''
Closeness Centrality (Mean shortest path length from v to all the other nodes in the network)
'''
'''
print("Closeness Centrality:")
V = [L[i][0] for i in range(len(L))]
D = {}

for i in range(311):
    D[V[i]] = nx.closeness_centrality(G, V[i])
    L = sorted(D.items(), key=lambda item: item[1], reverse=True)
for i in range(5):
    print(L[i][0], ":", L[i][1])
	
print()
'''


'''
The performance time in seconds will be printed out here.
'''
end = time.time()
print("Time in seconds", end - start)

def create_director_movie_rating_dictionaire(G):
	directors = []
	ratings = []
	movies = []
	genre = []
	year = []
	runtime = []
	director_active = []
	movie_active = []
	genre_active = []
	metascore = []
	gross = []
	sorted = []
	
	for m in G.edges.data():
		q = m[2]['movies']['Genre'].values
		n = m[2]['movies']['Director'].values
		o = m[2]['movies']['IMDB_Rating'].values
		p = m[2]['movies']['Series_Title'].values
		r = m[2]['movies']['Released_Year'].values
		s = m[2]['movies']['Runtime'].values
		t = m[2]['movies']['Meta_score'].values
		u = m[2]['movies']['Gross'].values
						
		for q2 in q:
			q3 = q2.split(', ')	
			for q4 in q3:
				for n1, o1, p1, r1, s1, t1, u1 in zip(n, o, p, r, s, t, u):
					directors.append(n1)
					ratings.append(o1)
					genre.append(q4)
					movies.append(p1)
					director_active.append(True)
					movie_active.append(True)
					genre_active.append(True)
					year.append(r1)
					runtime.append(s1)
					sorted.append(False)
					
					if t1 != -1:
						metascore.append(t1)
					else:
						metascore.append(0)
					
					if u1 != -1:
						gross.append(int(u1.split(',')[0]))
					else:
						gross.append(0)

	_dic = pd.DataFrame({'Genre': genre, 'Director': directors,'IMDB_Rating':ratings, 'Year':year, 'Movies': movies, 'Director_Act' : director_active, 'Movie_Act' : movie_active, 'Genre_Act' : genre_active, 'Sorted' : sorted, 'Runtime':runtime, 'Meta_Score':metascore, 'Gross':gross})
	
	return _dic
	
dic_data = create_director_movie_rating_dictionaire(G)

def get_movies_by_director(director, indices, dic_data):
	m = dic_data[['Director','Movies', 'Year', 'IMDB_Rating', 'Movie_Act', 'Director_Act', 'Genre_Act', 'Sorted']]
	m.loc[indices, 'Sorted'] = True

	a = []
	for i, j in m.iterrows():
		if director in j['Director']:
			if (j['Movie_Act']==True and j['Director_Act']==True and j['Genre_Act']==True and j['Sorted'] == True):
				a.append(i)
	
	m = m.loc[a].sort_values(by=['Movies'], ascending=False)
	return m	
	
def get_indices(dic_data, num_movies):	
	dic_data.loc[:, ['Sorted']] = False
	dic_data_copy = dic_data
	dic_genre = {'Genre' : [], 'Movies' : [], 'Director' : [], 'Count' : [], 'Meta_score' : [], 'Gross' : [], 'Director_Act' : [], 'Movie_Act' : [], 'Genre_Act' :[]}

	for g, mo, dir, m, gr, da, ma, ga in zip(dic_data_copy['Genre'], dic_data_copy['Movies'], dic_data_copy['Director'], dic_data_copy['Meta_Score'], dic_data_copy['Gross'], dic_data_copy['Director_Act'], dic_data_copy['Movie_Act'], dic_data_copy['Genre_Act']):
		dic_genre['Genre'].append(g)
		dic_genre['Count'].append(1)
		dic_genre['Director'].append(dir)
		dic_genre['Meta_score'].append(np.mean(m))
		dic_genre['Movies'].append(mo)
		dic_genre['Gross'].append(gr)
		dic_genre['Director_Act'].append(da)
		dic_genre['Movie_Act'].append(ma)
		dic_genre['Genre_Act'].append(ga)

	dic_genre = pd.DataFrame(dic_genre)
	dic_genre = dic_genre.sort_values(by=['Genre'], ascending=True)
	dic_genre = dic_genre.drop_duplicates()
	
	dic_data_copy = dic_data_copy[['Director', 'Movies', 'IMDB_Rating', 'Director_Act', 'Genre_Act', 'Movie_Act']].drop_duplicates()
	director_count = dic_data_copy.Director.value_counts()
	dic_genre = dic_genre.loc[dic_genre["Director"].isin(director_count[director_count >= num_movies].index)]
	directors_data = dic_data_copy.loc[dic_data_copy["Director"].isin(director_count[director_count >= num_movies].index)]
	dic_data.loc[dic_data['Director'].isin(directors_data['Director'].values), ['Sorted']] = True
	
	return 	dic_data.loc[dic_data['Director'].isin(directors_data['Director'].values), ['Sorted']].index

def avg_movie_rating(dic_data, num_movies):
	dic_data_copy = dic_data
	dic_genre = {'Genre' : [], 'Movies' : [], 'Director' : [], 'Count' : [], 'Meta_score' : [], 'Gross' : [], 'Director_Act' : [], 'Movie_Act' : [], 'Genre_Act' :[]}

	for g, mo, dir, m, gr, da, ma, ga in zip(dic_data_copy['Genre'], dic_data_copy['Movies'], dic_data_copy['Director'], dic_data_copy['Meta_Score'], dic_data_copy['Gross'], dic_data_copy['Director_Act'], dic_data_copy['Movie_Act'], dic_data_copy['Genre_Act']):
		dic_genre['Genre'].append(g)
		dic_genre['Count'].append(1)
		dic_genre['Director'].append(dir)
		dic_genre['Meta_score'].append(np.mean(m))
		dic_genre['Movies'].append(mo)
		dic_genre['Gross'].append(gr)
		dic_genre['Director_Act'].append(da)
		dic_genre['Movie_Act'].append(ma)
		dic_genre['Genre_Act'].append(ga)

	dic_genre = pd.DataFrame(dic_genre)
	dic_genre = dic_genre.sort_values(by=['Genre'], ascending=True)
	dic_genre = dic_genre.drop_duplicates()
	
	dic_data_copy = dic_data_copy[['Director', 'Movies', 'IMDB_Rating', 'Director_Act', 'Genre_Act', 'Movie_Act']].drop_duplicates()
	director_count = dic_data_copy.Director.value_counts()
	dic_genre = dic_genre.loc[dic_genre["Director"].isin(director_count[director_count >= num_movies].index)]
	directors_data = dic_data_copy.loc[dic_data_copy["Director"].isin(director_count[director_count >= num_movies].index)]
	directors_data = directors_data.loc[directors_data['Director_Act'] == True]
	directors_data = directors_data.loc[directors_data['Movie_Act'] == True]
	directors_data = directors_data.loc[directors_data['Genre_Act'] == True]
	
	dic_genre = dic_genre.loc[dic_genre['Director_Act'] == True]
	dic_genre = dic_genre.loc[dic_genre['Movie_Act'] == True]
	dic_genre = dic_genre.loc[dic_genre['Genre_Act'] == True]
		
	directors_data = directors_data.loc[directors_data["IMDB_Rating"] > -1]
	directors_data = directors_data.sort_values('Director')
	directors_data = directors_data[directors_data.IMDB_Rating != -1]
	number_movies = directors_data[['Director', 'Movies']].drop_duplicates()
	number_movies = number_movies.groupby('Director', as_index=False)
	number_movies = number_movies.count()
	directors_data = directors_data.groupby('Director', as_index=False)
	average_movies = directors_data.IMDB_Rating.mean()
	average_movies.insert(2, 'Nr_Movies', number_movies['Movies'], True)
	average_movies = average_movies.sort_values(by=['IMDB_Rating'], ascending=False)
	
	dic_genre.drop_duplicates()

	dic_genre_c = dic_genre.groupby('Genre', as_index = False).count()
	

	
	dic_genre_m = dic_genre.groupby('Genre', as_index = False).mean()
	dic_genre_m['Count'] = dic_genre_c['Count']
	

	dic_genre = dic_genre_m[['Genre', 'Count', 'Meta_score', 'Gross']]
	
	actors_list = sorted(list(G))

	g_x = []
	g_y = []
	g_size = []
	g_text = []
	g_genre = []
	g_metascore = []
	g_gross = []
	g_count = []

	for _genre, _count, _metascore, _gross in zip(dic_genre.Genre, dic_genre.Count, dic_genre.Meta_score, dic_genre.Gross):
		g_x.append(_gross)
		g_y.append(_metascore)
		g_size.append(20)
		g_count.append(_count)
		g_text.append('Genre: ' + str(_genre) + '<br>' + 'Movies: ' + str(int(_count)))
		g_genre.append(str(_genre))
		
	c_g_x = g_x	
	c_g_y = g_y	
	c_g_size = g_size

	genres_obj = pd.DataFrame({'Genre': g_genre,'X':g_x, 'Y': g_y, 'Size' : g_size,
	'Text' : g_text, 'C_X': g_x, 'C_Y': g_y, 'C_Size': g_size, 'Count': g_count})
	
	_hovertext = []
	for a, d, n in zip(average_movies['IMDB_Rating'], average_movies['Director'], average_movies['Nr_Movies']):
		_hovertext.append('AVG IMDB Rating: ' + str(a) + '<br>Director: ' + str(d) + '<br>Nr. of Movies: ' + str(n))
			
	return average_movies, _hovertext, genres_obj, actors_list
			
average_movies, _hovertext, genres_obj, actors_list = avg_movie_rating(dic_data, 1)	


app = dash.Dash()
		
app.layout = html.Div(children = [	
	html.Div(children = [
		html.Div(id='UI1', children=[
			dcc.Graph(id='network',
				figure=graph),
			], style={"display":"inline-block", "width" : "33%"}),
		html.Div(id='UI2', children=[
			dcc.Graph(id='avd',
				figure=go.Figure(go.Bar(
					x=average_movies['IMDB_Rating'].values,
					y=average_movies['Director'].values,
					hovertext=_hovertext,
					hoverinfo="text",
					orientation='h')).update_xaxes(title_text = "Average IMDB Rating of the Movies").update_yaxes(title_text = "Directors")),
		], style={"display":"inline-block", "width" : "33%"}),
		html.Div(id='UI3', children=[
			dcc.Graph(id='mg',
				figure=go.Figure(data=[go.Scatter(
						x=genres_obj['X'],
						y=genres_obj['Y'],
						text=genres_obj['Text'],
						mode='markers',
						visible = True,
						marker=dict(
							color=genres_obj['Count'].values,
							colorscale='ice_r',
							colorbar=dict(
								title="Nr. of Movies"
							),
							showscale=True,
							size=genres_obj['Size'].values,
							line=dict(
								color='Black',
								width=2
							)
						)
					)]).update_xaxes(
			title_text = "Average Gross").update_yaxes(
			title_text = "Average Meta Score")),	
		], style={"display":"inline-block", "width" : "33%"})
	], style={"display":"block"}),
	html.Div(children=[
		html.Div(children=[
			html.H3('First / Source Actor'),
			dcc.Dropdown(id="actor001", options=[
				{'label': i, 'value': i} for i in actors_list
				]),
			html.H3('Second / Target Actor'),
			dcc.Dropdown(id='actor002', options=[
				{'label': i, 'value': i} for i in actors_list
				]),
		], style={"display":"inline-block", "margin-right":"10px"}),
		html.Div(children=[
			html.H3('Graph Functions'),
			html.Button('Calculate Shortest Path', id='shortestpath', n_clicks=0),
			html.Button('Get Actor\'s degree', id='getDegree', n_clicks=0),
			html.Button('Get Connections', id='getNeighbors', n_clicks=0),
			html.Button('Get Movies', id='getMovies', n_clicks=0),
			html.Button('Remove Actor', id='rmActor', n_clicks=0),		
			html.Button('Add Actor', id='addActor', n_clicks=0),		
			html.Button('Set Default', id='setDefault', n_clicks=0),	
			html.Button('List Centralities', id='listCent', n_clicks=0)	
		], style={"display":"inline-block", "width":"14%", "margin-right":"10px"}),
		html.Div(children=[
			html.H3('Deselection of Directors'),
			dcc.Dropdown(id='deselected-directors',
				options=[
					{'label': i, 'value': i} for i in average_movies['Director'].sort_values(ascending=True).values
					],
				value=[],
				multi=True),
			html.H3('Deselection of Movies'),
			dcc.Dropdown(id='deselected-movies',
				options=[
					{'label': i, 'value': i} for i in dic_data['Movies'].drop_duplicates().sort_values(ascending=True).values
					],
					value=[],
			multi=True)	
		], style={"display":"inline-block", "margin-right":"10px"}),
		
		html.Div(children=[
			html.H3('Deselection of Genres'),
			#dcc.Checklist(id='checklist_genre',
			#	options=[
			#		{'label': i, 'value':i} for i in genres_obj['Genre'].sort_values(ascending=True)
			#	], value=genres_obj['Genre'].sort_values(ascending=True)),
			dcc.Dropdown(id='deselected_genre',
				options=[
					{'label': i, 'value': i} for i in genres_obj['Genre'].sort_values(ascending=True)
					],
					value=[],
			multi=True),
				html.H3('Show Director Movie Ratings'),
				dcc.Dropdown(id='directors_movies',
				options=[
					{'label': i, 'value': i} for i in average_movies['Director'].sort_values(ascending=True).values
						]
				)			
		], style={"display":"inline-block", "margin-right":"10px"}),
		
		html.Div(children=[
			html.H3('Move Genre'),
			dcc.Dropdown(id='selected-movies',
				options=[
					{'label': i, 'value': i} for i in genres_obj['Genre']
						]),
			html.H3('Number of Directors Movies'),
						dcc.Input( 
								id="input_{}".format("number"),
								type="number",
								placeholder="input type {}".format("number"),
								value=1,
								min=1,
								max=9
							)
		], style={"display":"inline-block", "margin-right":"10px"}),
		html.Div(children=[
			html.Div(children=[
				html.H3('Nr. Nodes:'),
				html.Div(id="nr_nodes", children=[]),
			]),
			html.Div(children=[
				html.H3('Nr. Edges:'),
				html.Div(id="nr_edges", children=[])
			]),			
		], style={"display":"inline-block", "margin-right":"10px"}),
	], style={"display":"inline-flex"}),
	
	html.Div( 
		id='selected_directors'),
	html.Div(
		id='info-director-movies'),
	html.Div( 
			id='selected_text'
			),
	html.Div( 
			id='info_movies'
			),

], style={'display':'block'})

	
def isCheckBoxActive(genre, value):
	i = 0
	for v in value:
		if v == genre:
			i = 1
	
	if i == 1:
		checkBox(genre)
	else:
		uncheckBox(genre)

def uncheckBox(genre):
	index = genres_obj.loc[genres_obj['Genre'] == genre].index.values[0]
	
	genres_obj.at[index, 'Size'] = 0
	genres_obj.at[index, 'X'] = -1
	genres_obj.at[index, 'Y'] = -1
	
def checkBox(genre):
	index = genres_obj.loc[genres_obj['Genre'] == genre].index.values[0]
	
	genres_obj.at[index, 'Size'] = genres_obj.at[index, 'C_Size']
	genres_obj.at[index, 'X'] = genres_obj.at[index, 'C_X']
	genres_obj.at[index, 'Y'] = genres_obj.at[index, 'C_Y']
	
def remove_node(n):
	m = G.nodes[n]['pos']
	
	removed_nodes.append({n:n,"pos":m})
		
	for neighbor in G.neighbors(n):
		removed_movies.append({n: G.get_edge_data(n, neighbor)})
	
	G.remove_node(n)	
	
def add_node(_node, _pos):
	G.add_node(_node, pos=_pos)
	
	for m in removed_movies:
		if _node in m.keys():
			i1 = m[_node]
			
			for i2 in i1['movies'].values:
				G.add_edge(i2[10], i2[11], movies=i1['movies'])		

	for i in range(len(removed_movies)-1, -1, -1):
		if (_node in removed_movies[i].keys()):
			del removed_movies[i]
		
			
def redefine_traces():	
	edge_trace = go.Scatter(
		x=[],
		y=[],
		line=dict(width=1, color='#888'),
		hoverinfo='none',
		mode='lines')
		
	'''
	Defining an edge trace for plotting the edges.
	'''	
	for edge in G.edges():
		x0, y0 = G.nodes[edge[0]]['pos']
		x1, y1 = G.nodes[edge[1]]['pos']
		edge_trace['x'] += tuple([x0, x1, None])
		edge_trace['y'] += tuple([y0, y1, None])
	
	node_trace = go.Scatter(
	x=[],
	y=[],
	text=[],
	mode='markers',
	hoverinfo='text',
	marker=dict(
		showscale=True,
		colorscale='Oranges_r',
		reversescale=True,
		color=[],
		size=15,
		colorbar=dict(
			thickness=10,
			title='Centrality of the leading actors',
			xanchor='left',
			titleside='right'
		),
		line=dict(width=0)))
	'''
	Add node positions to the node trace.
	'''				
	for node in G.nodes():
		x, y = G.nodes[node]['pos']
		node_trace['x'] += tuple([x])
		node_trace['y'] += tuple([y])
				
	'''
	Add all information about each node to the node trace.
	'''			
	for node, adjacencies in enumerate(G.adjacency()):
		node_trace['marker']['color']+=tuple([len(adjacencies[1])])
		node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
		node_trace['text']+=tuple([node_info])

	return edge_trace,node_trace

def getMoviesByGenre(genre, indices, dic_data):
	
	movies = dic_data[['Genre', 'Movies', 'Year', 'Runtime', 'Director', 'Movie_Act', 'Director_Act', 'Genre_Act', 'Sorted']]
	movies.loc[indices, 'Sorted'] = True
	
	a = []
	for i, j in movies.iterrows():
		if genre == j['Genre']:
			if (j['Movie_Act'] == True and 
				j['Director_Act'] == True and
				j['Genre_Act'] == True and j['Sorted'] == True):
				
				a.append(i)
	
	movies = movies.loc[a]
	
	return movies.drop_duplicates()
	
def focus_directors(dirs):
	options = []
	for i, d in dic_data.iterrows():
		if d['Director'] in dirs:
			dic_data.at[i, 'Director_Act'] = True
		else:
			dic_data.at[i, 'Director_Act']= False
			options.append(d['Director'])
	return list(set(options))
	
def focus_genres(genres):
	options = []
	for i, d in dic_data.iterrows():
		if d['Genre'] in genres:
			dic_data.at[i, 'Genre_Act'] = True
		else:
			dic_data.at[i, 'Genre_Act']= False
			options.append(d['Genre'])
	return list(set(options))	

def rm_node_from_list(node):
	for i in range(len(removed_nodes)-1, -1, -1):
		if (node in removed_nodes[i].keys()):
			del removed_nodes[i]
		
	
@app.callback([Output('avd', 'figure'),
			   Output('mg', 'figure'),
			   Output('network', 'figure'),
			   Output('info-director-movies','children'),
			   Output('avd', 'clickData'),
			   Output('directors_movies', 'value'),
			   Output('selected-movies', 'options'),
			   Output('directors_movies', 'options'),
			   Output('deselected-directors', 'value'),
			   Output('avd', 'selectedData'),
			   Output('mg', 'selectedData'),
			   Output('mg', 'clickData'),
			   Output('deselected_genre', 'value'),
			   Output('network', 'clickData'),
			   Output('network', 'selectedData'),
			   Output('nr_nodes', 'children'),
			   Output('nr_edges', 'children')
			   ], 
			[Input("input_{}".format("number"), "value"),
			Input('avd', 'selectedData'),
			Input('mg', 'selectedData'),
			Input('avd','clickData'), 	
			Input('mg','clickData'),			
			Input('directors_movies', 'value'),
			Input('deselected-directors', 'value'),
			Input('deselected-movies', 'value'),
			Input('selected-movies', 'value'),
			Input('deselected_genre', 'value'),
			Input('network', 'selectedData'),
			Input('network','clickData'),
			Input('actor001', 'value'),
			Input('actor002', 'value'),
			Input('shortestpath', 'n_clicks'),
			Input('getDegree', 'n_clicks'),
			Input('getNeighbors', 'n_clicks'),
			Input('getMovies', 'n_clicks'),
			Input('rmActor', 'n_clicks'),
			Input('setDefault','n_clicks'),
			Input('addActor', 'n_clicks'),
			Input('listCent', 'n_clicks')])
def create_new_avd(number_movies_AVD, selectedDataAVD, selectedDataMG, clickDataAVD, clickDataMG, dir_movies_value, des_dir_value, des_movie_value, sel_movies_MG,
					deselect_genre_MG, selectedDataNet, clickDataNet, actor001, actor002, btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8):	
	
	node_name = ''
	genre_opt = []
	dir_opt = []
	c = ''
	movies_mg = []
						
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'shortestpath' in changed_id:
		if actor001 != None and actor002 != None:

			try:
				j = nx.shortest_path(G, source=actor001, target=actor002)
								
				m = ''
				
				for i1, i2 in zip(range(len(j)), j):
					if i1 < (len(j) - 1):
						m += i2 + ' - '
					else:
						m += i2
				c = [html.H1('Shortest path between ' + str(actor001) + ' and ' + str(actor002) + ': '), 
					 html.H2(m)]				
			except:
				c = [html.H1('No connection path found in the system.')]				
		else:
			if actor001 == None:
				c = [html.H1('First actor is empty.')]			
			if actor002 == None:
				c = [html.H1('Second actor is empty.')]			

	elif 'getDegree' in changed_id:
		if actor001 != None:
			c = [html.H1(str(actor001) + ' has ' + str(G.degree(actor001)) + ' degree')]			
		else:
			c = [html.H1('First actor is empty')]			
	elif 'getNeighbors' in changed_id:
		if actor001 != None:
			try:
				m = ''
				cneighbors = 0
				for neighbor in G.neighbors(actor001):
					cneighbors += 1
					m += str(neighbor) + ', '
					
				c = [html.H1(str(cneighbors) + ' neighbors found: '),
					 html.H2(str(m))]			
			except:
				c = [html.H1('Actor is not in the graph.')]
		else:
			c = [html.H1('First actor is empty.')]			
	elif 'getMovies' in changed_id:
		if actor001 != None and actor002 != None:
			try:
				cmovies = 0
				movies = ''
				for star1, star2, movielist, in G.edges(data=True):
					if ((star1 == actor001 and star2 == actor002) or
					   (star2 == actor001 and star1 == actor002)):
					   
							for m in movielist['movies']['Series_Title']:
								cmovies += 1
								movies += str(m) + ', '
					   
				c = [html.H1(str(cmovies) + ' movies found: '), html.H2(str(movies))]			
			except:
				c = [html.H1('First or second actor is empty')]			
		else:
			c = [html.H1('First or second actor is empty')]			
	elif 'rmActor' in changed_id:
		if actor001 != None:	
			try:
				n = actor001
				node_name = n
				remove_node(n)
				c = [html.H1('Deleted: ' + str(n))]			
			except:
				c = [html.H1(str(n) + ' is already removed')]			
	elif 'addActor' in changed_id:
		if actor001 != None:
			try:
				n = actor001
				node_name = n
				nodec = 0
				for i, (k, v) in zip(range(len(removed_nodes)), removed_nodes):
					if k == n:
						nodec += 1
						add_node(removed_nodes[i][k], removed_nodes[i]['pos'])	
				rm_node_from_list(n)
				
				if nodec != 0:
					c = [html.H1('Added: ' + str(n))]			
				else:
					c = [html.H1(str(n) + ' is already in the system.')]			
			except:
				c = [html.H1(str(n) + ' is already in the system.')]			

	elif 'setDefault' in changed_id:
		for i, (k, v) in zip(range(len(removed_nodes)), removed_nodes):
			add_node(removed_nodes[i][k], removed_nodes[i]['pos'])
		removed_nodes.clear()	
	elif 'listCent' in changed_id:
		m = get_descending_degree_centralities_of_stars()
		
		c = [html.H1(str(len(m)) + ' entries found:'), dash_table.DataTable(
			id='table-dash',
			columns=[{"name": i, "id": i} for i in m.columns],
			data=m.to_dict('records'),
			editable=True,
			row_selectable="multi",
			selected_rows=[],
			page_action="native",
			page_current= 0,
			page_size= 10,
			style_data={
				'width': '100px',
				'maxWidth': '200px',
				'minWidth': '200px',
			},

			style_table={
				'overflowY': 'scroll',
				'height': 200,
			})]
				
	selectedDataNet = [i['text'].split(' #')[0] for i in selectedDataNet['points']] if selectedDataNet else None
	clickDataNet = [i['text'].split(' #')[0] for i in clickDataNet['points']] if clickDataNet else None
	
	if selectedDataNet != None:
		try:
			rm_nodes = []
			
			for n in G.nodes():
				if n not in selectedDataNet:
					rm_nodes.append(n)
					
			for n in rm_nodes:
				remove_node(n)
		except nx.NetworkXNoPath:
			print('Nothing to remove')
		
	edge_trace, node_trace = redefine_traces()
	dic_data = create_director_movie_rating_dictionaire(G)

	indices = get_indices(dic_data, number_movies_AVD)					
	
	graph = go.Figure(data=[edge_trace, node_trace],
					 layout=go.Layout(
						title='',
						titlefont=dict(size=16),
						showlegend=False,
						hovermode='closest',
						margin=dict(b=20,l=5,r=5,t=40),
						annotations=[dict(
							text="",
							showarrow=False,
							xref="paper", yref="paper")
							],
						xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
						yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))					
	graph.update_layout({'plot_bgcolor': 'rgb(137, 207, 237)'})

	for i, d in dic_data.iterrows():
		if d['Director'] in des_dir_value:
			dic_data.at[i, 'Director_Act'] = False
		else:
			dic_data.at[i, 'Director_Act']= True	
			
	for i, d in dic_data.iterrows():
		if d['Genre'] in deselect_genre_MG:
			dic_data.at[i, 'Genre_Act'] = False
		else:
			dic_data.at[i, 'Genre_Act']= True	
	
	for i, d in dic_data.iterrows():
		if d['Movies'] in des_movie_value:
			dic_data.at[i, 'Movie_Act'] = False
		else:
			dic_data.at[i, 'Movie_Act']= True	
	'''
	for i, d in dic_data[['Genre', 'Genre_Act']].drop_duplicates().iterrows():
		if d['Genre_Act'] == True:
			genre_opt.append({'label': d['Genre'], 'value': d['Genre']})
			
	for i, d in dic_data[['Director', 'Director_Act']].drop_duplicates().iterrows():
		if d['Director_Act'] == True:
			dir_opt.append({'label': d['Director'], 'value':d['Director']})						
	'''					
	if selectedDataAVD != None:
		selectedDataAVD = [i['label'] for i in selectedDataAVD['points']] if selectedDataAVD else None
						
		des_dir_value = focus_directors(selectedDataAVD)
		
		dir_opt = []
		for i, d in dic_data[['Director', 'Director_Act']].drop_duplicates().iterrows():
			if d['Director_Act'] == True:
				dir_opt.append({'label': d['Director'], 'value':d['Director']})
			
	if (selectedDataMG != None):	
		selectedDataMG = [i['text'].split('<br>')[0] for i in selectedDataMG['points']] if selectedDataMG else None
		selectedDataMG = [i.split('Genre: ')[1] for i in selectedDataMG] if selectedDataMG else None
		
		deselect_genre_MG = focus_genres(selectedDataMG)
		
		genre_opt = []
		for i, d in dic_data[['Genre', 'Genre_Act']].drop_duplicates().iterrows():
			if d['Genre_Act'] == True:
				genre_opt.append({'label': d['Genre'], 'value': d['Genre']})
			
	if (clickDataMG != None or sel_movies_MG != None):
		if sel_movies_MG == None:
			clickDataMG = [i['text'].split('<br>')[0] for i in clickDataMG['points']] if clickDataMG else None
			clickDataMG = [i.split('Genre: ')[1] for i in clickDataMG] if clickDataMG else None
			clickDataMG = clickDataMG[0]
		elif sel_movies_MG != None:
			clickDataMG = sel_movies_MG
		
		movies_mg = getMoviesByGenre(clickDataMG, indices, dic_data)
		movies_mg = movies_mg[['Movies', 'Year', 'Runtime', 'Director']]
		movies_mg = movies_mg.sort_values(by=['Movies'], ascending=True)
		
		c = [html.H1(clickDataMG  +  ' - ' + str(len(movies_mg)) + ' entries found:'), 
			dash_table.DataTable(
			id='table',
			columns=[{"name": i, "id": i} for i in movies_mg.columns],
			data=movies_mg.to_dict('records'),
			style_data={
				'width': '100px',
				'maxWidth': '200px',
				'minWidth': '200px',
			},
			style_table={
				'overflowY': 'scroll',
				'height': 100,
			})]
	
	if clickDataAVD != None or dir_movies_value != None:
		
		if dir_movies_value == None:
			clickDataAVD = [i['label'] for i in clickDataAVD['points']] if clickDataAVD else None
			clickDataAVD = clickDataAVD[0]	
		elif dir_movies_value != None:
			clickDataAVD = dir_movies_value
				
		m = get_movies_by_director(clickDataAVD, indices, dic_data)		
		m = m[['Movies', 'Year', 'IMDB_Rating']].drop_duplicates()
		m = m.sort_values(by=['Movies'], ascending=True)

		c = [html.H1(clickDataAVD +  ' - ' + str(len(m)) + ' entries found:'), dash_table.DataTable(
			id='table-dash',
			columns=[{"name": i, "id": i} for i in m.columns],
			data=m.to_dict('records'),
			editable=True,
			row_selectable="multi",
			selected_rows=[],
			page_action="native",
			page_current= 0,
			page_size= 10,
			style_data={
				'width': '100px',
				'maxWidth': '200px',
				'minWidth': '200px',
			},

			style_table={
				'overflowY': 'scroll',
				'height': 200,
			})]
		
	dir_opt = []
	genre_opt = []
	
	for i, d in dic_data.iterrows():
		if d['Genre'] in deselect_genre_MG:
			dic_data.at[i, 'Genre_Act'] = False
		else:
			dic_data.at[i, 'Genre_Act']= True	
	
	for i, d in dic_data.iterrows():
		if d['Movies'] in des_movie_value:
			dic_data.at[i, 'Movie_Act'] = False
		else:
			dic_data.at[i, 'Movie_Act']= True	
	
	for i, d in dic_data.iterrows():
		if d['Director'] in des_dir_value:
			dic_data.at[i, 'Director_Act'] = False
		else:
			dic_data.at[i, 'Director_Act']= True	

	for i, d in dic_data[['Genre', 'Genre_Act']].drop_duplicates().iterrows():
		if d['Genre_Act'] == True:
			genre_opt.append({'label': d['Genre'], 'value': d['Genre']})
			
	for i, d in dic_data[['Director', 'Director_Act']].drop_duplicates().iterrows():
		if d['Director_Act'] == True:
			dir_opt.append({'label': d['Director'], 'value':d['Director']})

	average_movies, _hovertext, genres_obj, actors_list = avg_movie_rating(dic_data, number_movies_AVD)	

	dir_chart = go.Figure(go.Bar(
				x=average_movies['IMDB_Rating'].values,
				y=average_movies['Director'].values,
				hovertext=_hovertext,
				hoverinfo="text",
				orientation='h'))
				
	dir_chart.update_xaxes(title_text = "Average IMDB Rating of the Movies")			
	dir_chart.update_yaxes(title_text = "Directors")
	dir_chart.update_layout(hovermode ='y unified')
	
	bchart = go.Figure(data=[go.Scatter(
				x=genres_obj['X'].values, y=genres_obj['Y'].values,
				text=genres_obj['Text'].values,
				mode='markers',
				visible = True,
				marker=dict(
					color=genres_obj['Count'].values,
					colorscale='ice_r',
					colorbar=dict(
						title="Nr. of Movies"
					),
					showscale=True,
					size=genres_obj['Size'].values,
					line=dict(
						color='Black',
						width=2
					)
				)
			)])
			
	bchart.update_xaxes(title_text = "Average Gross")
	bchart.update_yaxes(title_text = "Average Meta Score")
					
	return dir_chart, bchart, graph, c, None, None, genre_opt, dir_opt, des_dir_value, None, None, None, deselect_genre_MG, None, None, html.H3(str(nx.number_of_nodes(G))), html.H3(str(nx.number_of_edges(G)))
				
if __name__ == '__main__':
	app.run_server()