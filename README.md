# IMDB-Data-Analysis-and-Visualization-App

## Table of Contents  
- [Introduction](#introduction) 
- [Features](#features)
- [Usage](#usage)
- [Notes](#notes)
- [Future](#future)
- [References](#references)

## Introduction
I developed this app as part of the Master's course [Computational and Visual Network Analsysis](https://lnu.se/en/research/PhD-studies/kurser/ftk/computational-and-visual-network-analysis/) of the study program computer science at [Linnaeus University](https://lnu.se/) in Växjö, Sweden. The goal of this app is to read IMDB data from imdb_top_1000.csv, filter it and visualize it in parallel. This app was developed with Python 3.7. The Python frameworks NetworkX and Plotly were used for the graphical representation and Dash was used to dynamically manipulate the data. 

Three graphical diagrams are presented: The first graph is a network of the first and second main actors who played in common movies.

The second graph is the representation of a horizontal bar chart of directors with the average IMDB of their entire movies.

The third graph is the representation of movie genres with their average meta score and average one-player bites. The brightness of the color determines the number of movies assigned to the genre.

<img alt="Centrality of the leading actors" src="https://github.com/RamoramaInteractive/IMDB-Data-Analysis-and-Visualization-App/blob/main/Screenshots/Screen001.jpg" width="30%" height="30%"><img alt="Average IMDB Rating of the Movies" src="https://github.com/RamoramaInteractive/IMDB-Data-Analysis-and-Visualization-App/blob/main/Screenshots/Screen002.jpg" width="30%" height="30%"><img alt="Average Gross of the Movie Genres" src="https://github.com/RamoramaInteractive/IMDB-Data-Analysis-and-Visualization-App/blob/main/Screenshots/Screen003.jpg" width="30%" height="30%">

## Features
* Calculation of the shortest path between two leading actors.
* Determination of an actor's degree (number of connections).
* Listing the connections and movies of an certain actor.
* Addition, removal, selecting and deselecting of certain actors.
* Setting the modifications of the actor's network back to default.
* Listing the centralities of an certain actor (rankings of actors within the network graph corresponding to their network position.).
* Selection and Deselection of directors, movies and genre.
* Listing of movie genres, ratings of the director's movies and their ratings.

## Usage
Before you start the application, make sure you install the required libraries from requirement.txt.

`pip install -r requirements.txt`

To start the application you need to type the following

`python ramon_wilhelm_assignment003.py`.

If you see a message like

`Dash is running on http://127.0.0.1:8050/`

Copy the IP address and paste it into your browser. 

## Notes
Due to the Spring layout and the calculation of the actor network, there are slight delays. If the network is reduced, the app runs faster. Due to the filtering of the movies, the actor network does not change. 

## Future
Improvements and bug fixing are planned.

## References
* [IMDB Movie Dataset](https://www.kaggle.com/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
* [NetworkX](https://networkx.org/documentation/stable/reference/algorithms/index.html)
* [Plotly](https://plotly.com/)
* [Dash](https://dash.plotly.com/)
