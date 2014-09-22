fantasy-football-drafter
========================

Fantasy football draft analysis and tools for setting up your own draft.

This is a project designed to help optimize and personalize the fantasy
football draft experience.  It includes a set up functions to analyze
position performance in previous seasons (2009-2013) and compare the
average draft position at the start of the season to the actual
performance at the end of the season. The functions are designed to
allow users to inspect data and come up with their own model for how to
set the expected value of each player during the draft. In addition to
the analysis functions there is a stand-alone python script that sets up
a snaking draft incorporating each users pre-defined custom model that
will make recommendations during the draft.



Getting started:

To get started you'll need the nfldb from BurntSushi:
https://github.com/BurntSushi/nfldb

Install the Postgres database following instructions on the BurntSushi
github.  In these python scripts the database is named nfldb and user is
ffuser with no password.  You will also need to make sure you have the
psycopg2 python wrapper for accessing Postgres.


ffmodel.py
==========

ffmodel contains a series of functions that can be used to explore the
performance of players in previous seasons.  The default fantasy point
conversations are initialized at the start of the script and should be
adjusted for your league (this is default for Yahoo).

Most of the functions are helper functions for the two main functions:
1) createADPdf This function scrapes
http://football.myfantasyleague.com/ for average draft position for the
years 2009-2013.  It returns a dataframe with columns for the player
name, position, team, rank (order by position), ADP (average draft
position), and year.  This will be used as the preseason predictor for
performance.


2) PtsvPick This function takes as input the dataframe from createADPdf
and computes the actual fantasy points scored by each player in each
season using the local nfldb. Option arguments allow the user to specify
years and the number of players at each position.

The remaining functions are for fitting curves to the predicted vs
actual performance, plotting, and predicting future performance given a
rank and position.

drafttools.py
=============

drafttools defines two classes: a Draft class and a Model class. These
are used to run a snaking draft using pre-defined custom models for how
to order players based on value over replacement and slots remaining.

To begin, you will need to download a csv file from
http://www.fantasypros.com/nfl/rankings (You might have to pretty up the
headers) If you're starting the season, you'll want the preseason
rankings for all positions out to at least 250.

The model takes input: -type (defaults to 'VOR' or value over
replacement, otherwise 'Rank' or 'ADP' just uses the input data file in
without computing value over replacement) -auto (bool value that tells
drafting tool to automatically pick for you) -data (the csv file loaded
as a dataframe of player ranks and positions) -draft positition (order
in the draft, 0 indexed) -nplayers (the number of managers in the draft)
-rounds (the number of rounds in the draft) -params (dictionary of the
parameters of the model, defined below)

Model parameters keys:

fitcol: Choose either 'ADP' or 'Rank'.  Order players in each position
by average draft position or pundit ranking.  Will comput VOR using this
ordering

for each position ('DST','K','QB','RB','WR','TE'): The model is the
designed to take the fit coefficients for previous years' fantasy points
vs preseason rank by position, as well as fits for fantasy points vs
actual postseason rank.

The rest of the parameters are position specific with the order:
'QB','RB','WR','TE','K','DST' fitweight: The user can weight toward a
fit on the preseason rank (a vote for average confidence in
predictability of performance, value of 0) or toward postseason rank (a
vote for high predictability in performance value of 1) at each position
by controlling the fit weight value.  See fit functions in ffmodel for
details.

posexp: The number of desired players at each position.

denom: Used when adjusting the VOR based off of available positions. 
The adjusted VOR is: VOR * posexp/denom Setting denom to posexp equally
weights all positions at the start of the draft. It is recommended to
set denom high for K and DST to draft them last.

VORper: When computing VOR, at what percentile do you define the
replacement player? For example, to get the VOR for a quarterback, you
compute his value, then subtract the value of a replacement quarterback.
 Setting this toward the low (10-25th) percentile is recommended. 
Higher setting run the risk of drafting into negative VOR which will
mess with your position adjustment since this is a divisive adjustment. 
That is, if you are in the middle of the draft and likely wanting to
draft a running back, but are past the replacement position, all running
backs will have negative VOR which is larger than the undrafted Kickers
and defense that have had their VORs squished toward 0.

Running a draft 
=============== 
Use the testdraft.py file as a template.
 For each player, initialize a model with the chosen parameters and
draft order.  Initialize the draft with the list of models for each
manager.  Run the script which will output a csv file at the end. Beer +
pizza is recommended.