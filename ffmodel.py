'''
Fantasy Football Model
'''
import pandas as pd
import numpy as np
import psycopg2 as psdb
import urllib2
from bs4 import BeautifulSoup as bs
from matplotlib import pyplot as plt

# database authentication
dbauth = {'host':'localhost',
          'user':'ffuser',
          'dbname':'nfldb'}

# default point model as an array for individual players
# these are point per:
# pass yd, pass td, int, rush yd, rush td, rec yd, rec td, return td, 2pt conv, fumble
pm_i = np.array([1./25.,4,-1,0.1,6,0.1,6,6,2,-2])

# default point model as an array for kickers
# these are point per:
# fg0-39,fg40-49,fg50+,pat
pm_k = np.array([3,4,5,1])

# default point model as an array for defense
# these are point per:
# sack,int,fumble recovery,td,safety,block kick, kickoff/punt return td,
# shutout, 1-6,7-13,14-20,21-27,28-34,35+ pts allowed
pm_d = np.array([1,2,2,6,2,2,6,10,7,4,1,0,-1,-4])

def dbconn():
    return psdb.connect(**dbauth)

def getPlayerStats(player,yr,pos,pointmodel=pm_i):
    # return a fantasy points per season
    qry = '''
    SELECT SUM(pp.passing_yds),
    SUM(pp.passing_tds),
    SUM(pp.passing_int),
    SUM(pp.rushing_yds),
    SUM(pp.rushing_tds),
    SUM(pp.receiving_yds),
    SUM(pp.receiving_tds),
    SUM(pp.puntret_tds)+SUM(pp.kickret_tds),
    SUM(pp.passing_twoptm)+SUM(pp.rushing_twoptm)+SUM(pp.receiving_twoptm),
    SUM(pp.fumbles_tot)
    FROM play_player pp
    JOIN game ON
    pp.gsis_id=game.gsis_id
    JOIN player p ON
    p.player_id=pp.player_id
    WHERE game.season_type='Regular' AND game.season_year={0}
    AND p.full_name = '{1}' AND p.position='{2}'
    '''.format(yr,player,pos)
    con=dbconn()
    with con:
        df=pd.io.sql.read_frame(qry,con)
    # make into np array and scale by point model
    ptarray=np.array(df)
    try:
        pts=(ptarray*pointmodel).sum()
    except:
        pts=0
    return pts

def getKickerStats(player,yr,pointmodel=pm_k):
    qry = """
    SELECT 
    CASE
      WHEN pp.kicking_fgm_yds>0 AND pp.kicking_fgm_yds<40 THEN 'short'
      WHEN pp.kicking_fgm_yds>=40 AND pp.kicking_fgm_yds<50 THEN 'medium'
      WHEN pp.kicking_fgm_yds>=50 THEN 'long'
      WHEN pp.kicking_xpa=1 THEN 'pta'
    END as pointtype,
    COUNT(*) as total
    FROM play_player pp
    JOIN game ON
    pp.gsis_id=game.gsis_id
    JOIN player p ON
    p.player_id=pp.player_id
    WHERE game.season_type='Regular' AND game.season_year={0}
    AND p.full_name = '{1}' AND p.position IN ('K', 'UNK')
    GROUP BY 1
    """.format(yr,player)
    con=dbconn()
    with con:
        df=pd.io.sql.read_frame(qry,con)
    # make into np array and scale by point model
    pts=0
    if df.total[df.pointtype=='short']:
        pts+=np.int(df.total[df.pointtype=='short'])*pointmodel[0]
    if df.total[df.pointtype=='medium']:
        pts+=np.int(df.total[df.pointtype=='medium'])*pointmodel[1]
    if df.total[df.pointtype=='long']:
        pts+=np.int(df.total[df.pointtype=='long'])*pointmodel[2]
    if df.total[df.pointtype=='short']:
        pts+=np.int(df.total[df.pointtype=='pta'])*pointmodel[3]
    return pts

def getDefenseStats(team,yr,pointmodel=pm_d):
    qry='''
    SELECT
    CASE
      WHEN (home_team='{0}' AND away_score=0) THEN 'shutout'
      WHEN (home_team='{0}' AND away_score>=1 AND away_score<=6) THEN 'p1'
      WHEN (home_team='{0}' AND away_score>=7 AND away_score<=13) THEN 'p7'
      WHEN (home_team='{0}' AND away_score>=14 AND away_score<=20) THEN 'p14'
      WHEN (home_team='{0}' AND away_score>=21 AND away_score<=27) THEN 'p21'
      WHEN (home_team='{0}' AND away_score>=28 AND away_score<=34) THEN 'p28'
      WHEN (home_team='{0}' AND away_score>=35) THEN 'p35'
      WHEN (away_team='{0}' AND home_score=0) THEN 'shutout'
      WHEN (away_team='{0}' AND home_score>=1 AND home_score<=6) THEN 'p1'
      WHEN (away_team='{0}' AND home_score>=7 AND home_score<=13) THEN 'p7'
      WHEN (away_team='{0}' AND home_score>=14 AND home_score<=20) THEN 'p14'
      WHEN (away_team='{0}' AND home_score>=21 AND home_score<=27) THEN 'p21'
      WHEN (away_team='{0}' AND home_score>=28 AND home_score<=34) THEN 'p28'
      WHEN (away_team='{0}' AND home_score>=35) THEN 'p35'
    END AS ptsa, COUNT(*) as total
    FROM game
    WHERE season_type='Regular' AND season_year={1}
    AND (away_team='{0}' OR home_team='{0}')
    GROUP BY 1
    '''.format(team,yr)
    con=dbconn()
    with con:
        dfgame=pd.io.sql.read_frame(qry,con)
   
    # get player specific stats
    qry2 = """
    SELECT SUM(pp.defense_sk) as sacks,
    SUM(pp.defense_int) as ints,
    SUM(pp.defense_frec) as fumr,
    SUM(pp.defense_frec_tds + pp.defense_int_tds) as dtds,
    SUM(pp.defense_safe) as safe,
    SUM(pp.defense_puntblk + pp.defense_fgblk) as blks,
    SUM(pp.defense_misc_tds) as dtdsother
    FROM play_player pp
    JOIN game ON
    pp.gsis_id=game.gsis_id
    JOIN player p ON
    p.player_id=pp.player_id
    WHERE game.season_type='Regular' AND game.season_year={0}
    AND p.team = '{1}'
    """.format(yr,team)
    con=dbconn()
    with con:
        dfplayer=pd.io.sql.read_frame(qry2,con)
    # combine and tally
    ptarray=np.zeros_like(pointmodel)
    playerarr=np.array(dfplayer)[0]
    
    for i in xrange(7):
        ptarray[i]=playerarr[i]*pointmodel[i]
    if dfgame.total[dfgame.ptsa=='shutout']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='shutout'])*pointmodel[7]
    if dfgame.total[dfgame.ptsa=='p1']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p1'])*pointmodel[8]
    if dfgame.total[dfgame.ptsa=='p7']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p7'])*pointmodel[9]
    if dfgame.total[dfgame.ptsa=='p14']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p14'])*pointmodel[10]
    if dfgame.total[dfgame.ptsa=='p21']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p21'])*pointmodel[11]
    if dfgame.total[dfgame.ptsa=='p28']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p28'])*pointmodel[12]
    if dfgame.total[dfgame.ptsa=='p35']:
        ptarray[7]=np.int(dfgame.total[dfgame.ptsa=='p35'])*pointmodel[13]
    return ptarray.sum()

def makeRatingDF():
    '''
    create a dataframe of each player/team, position, and fantasy points for each year
    along with some additional info
    '''
    qry="""
    SELECT player_id,full_name,position,birthdate,years_pro
    FROM player
    """
    con=dbconn()
    with con:
        df=pd.io.sql.read_frame(qry,con)
    for yr in [2009,2010,2011,2012,2013]:
        df[str(yr)]=0
    for indx,row in df.iterrows():
        pos=row['position']
        name=row['full_name']
        # fix names with a ' in them
        if name!=None:
            name=name.replace("'","''")
        if pos in ['QB','RB','TE','WR']:
            for yr in [2009,2010,2011,2012,2013]:
                try:
                    pts=getPlayerStats(name,yr,pos,pm_i)
                except:
                    pts=0
                df[str(yr)][indx]=pts
        if pos=='K':
            for yr in [2009,2010,2011,2012,2013]:
                try:
                    pts=getKickerStats(name,yr,pm_k)
                except:
                    pts=0
                df[str(yr)][indx]=pts
    return df

def ageCurve(df,playername,pos):
    # plot aging curve for a player as normalized derivative of performance
    season=[2009,2010,2011,2012,2013]
    dseries=df[(df.full_name==playername)*(df.position==pos)]
    dob=dseries.birthdate.values[0]
    yrborn=int(dob.split('/')[2])
    performance=[]
    age=[]
    for s in season:
        pts=dseries[str(s)].values[0]
        if pts>0:
            age.append(s-yrborn)
            performance.append(pts)
    if len(performance)>1:
        performance=np.array(performance)
        # get derivative
        performance=(performance[1:]-performance[:-1])/(1.0*performance[:-1])
        age.pop(0)
    else:
        performace = []
        age = []
    return performance,age

def ageAnalysis(df,pos):
    # get age curve for all players of pos
    # data is a little too sparse for this
    subdf=df[df.position==pos]
    allp=[]
    alla=[]
    for row in subdf.iterrows():
        player=row[1]['full_name']
        p,a=ageCurve(subdf,player,pos)
        if len(p)>0:
            allp.extend(list(p))
            alla.extend(list(a))
    allp=np.array(allp)
    alla=np.array(alla)
    agerange=np.unique(alla)
    medp=[]
    for uage in agerange:
        medp.append(np.median(allp[alla==uage]))
    medp=np.array(medp)
    return medp,agerange

def getHistoricalADP(yr):
    # get the historical average draft position of a given year
    # using http://football.myfantasyleague.com/
    # pulls the top 250
    # returns the rank and average draft position along with player name, position and team
    # in a dataframe
    htmladdr='http://football.myfantasyleague.com/'+str(yr)+'/adp?COUNT=250'
    req = urllib2.Request(htmladdr)
    soup = bs(urllib2.urlopen(req).read())
    # parse rows from table
    allrows=soup.findAll('tr')
    datarows=allrows[3:-4] # get rid of headers and footers
    
    # got through and extract data
    data=[]
    for row in datarows:
        # order is rank, player info, avg draft position, min, max, and n for drafts
        rank = int(row.findAll('td')[0].get_text().replace('.',''))
        playerinfo=row.findAll('td')[1].get_text().replace(',','').strip().split()
        if len(playerinfo)==4:
            ln,fn,team,pos = playerinfo
            fullname=fn+' '+ln
        else: # probably a team or weird first/last name
            if str(playerinfo[-1])=='Def':
                # parse out backward
                pos = playerinfo[-1]
                team = playerinfo[-2]
                fn = playerinfo[1:-2] + [playerinfo[0]]
                fullname = ' '.join(fn)
        adp = float(row.findAll('td')[2].get_text())
        if pos=='PK':
            pos='K'
        data.append([fullname,str(pos),str(team),rank,adp,yr])
        
    df=pd.DataFrame(data,columns=('fullname','pos','team','rank','adp','yr'))
    return df

def createADPdf():
    dfall=[]
    for yr in [2009,2010,2011,2012,2013]:
        df=getHistoricalADP(yr)
        if len(dfall)>0:
            dfall = pd.concat([dfall,df])
        else:
            dfall=df
    return dfall


def PtsvPick(histadp,yrs=[2009,2010,2011,2012,2013],ns=[60,60,25,15,12,12]):
    # get the average performance in fantasy points as function of pre-season average
    # draft position averaged (median) across years 2009-2013 with MAD error
    # return expected points and error for each position and rank
    postype=['RB','WR','QB','TE','K','Def']
    
    datadict={}
    for i,p in enumerate(postype):
        # make a matrix to store performance vs ADP rank for each year
        n=ns[i]
        pvadp = np.zeros((len(yrs),n))
        adp=histadp[histadp.pos==p]
        # for each year
        for j,yr in enumerate(yrs):
            # get top n average draft picks by that position
            names=adp[adp.yr==yr].fullname # should be sorted by rank already
            # get each players actual performance
            for z,fn in enumerate(names):
                if z>=n:
                    break
                fn=fn.replace("'","''")
                # handle some exceptions
                if fn=='Chris Wells':
                    fn='Beanie Wells'
                if fn=='Chad Ochocinco':
                    fn='Chad Johnson'
                if fn=='Carnell Williams':
                    fn='Cadillac Williams'
                if fn=='Giovani Bernard':
                    fn='Gio Bernard'
                if fn=='Stevie Johnson':
                    fn='Steve Johnson'
                
                if p=='K':
                    pts=getKickerStats(fn,yr)
                elif p=='Def':
                    team=adp[adp.yr==yr].team.iloc[z]
                    if team in ['GBP','KCC','NEP','NOS','SDC','SFO','TBB']:
                        team=team[:-1]
                    pts=getDefenseStats(team,yr)
                else:
                    pts=getPlayerStats(fn,yr,p)
                    if fn=='Mike Tolbert':
                            pts=getPlayerStats(fn,yr,'FB')
                    if pts==0:
                        # probably a player that switched teams and has UNK as position
                        pts=getPlayerStats(fn,yr,'UNK')
                if pts==0:
                    print 'Could not find %s in %i'%(fn,yr)    
                pvadp[j,z]=pts
        # median averaging, and show ordered rank (of drafted players)
        if len(yrs)>1:
            datadict[p]=[np.median(pvadp,axis=0),np.median(np.abs(pvadp-(np.median(pvadp,axis=0),))),\
            np.median(-np.sort(-pvadp,axis=1),axis=0)]
            # mean averaging
            # datadict[p]=[np.mean(pvadp,axis=0),np.std(pvadp,axis=0),np.mean(-np.sort(-pvadp,axis=1),axis=0)]
        else:
            datadict[p]=[pvadp[0],0,-np.sort(-pvadp[0])]
    return datadict

def fitPtsvRank(points,fitfun='l'):
    # regress on points vs rank order
    rank=np.array([[np.ones(len(points))],[np.arange(len(points))]])
    rank=rank.reshape(2,len(points))
    points=np.array(points)
    points=points.reshape(len(points),1)
    # normal equation
    if fitfun=='l':
        theta=np.linalg.pinv(rank.dot(rank.T)).dot(rank.dot(points))
        yhat=rank.T.dot(theta)
    else: # exponential fit
        points=np.log(points)
        theta=np.linalg.pinv(rank.dot(rank.T)).dot(rank.dot(points))
        yhat = np.exp(theta[0])*np.exp(theta[1] * rank[1])
    return theta

def predictPtsvRank(theta,nranks,fitfun='l'):
    # predict points from fit coefficients out to nranks
    rank=np.array([[np.ones(nranks)],[np.arange(nranks)]])
    rank=rank.reshape(2,nranks)
    if fitfun=='l':
        yhat=rank.T.dot(theta)
    else: # exponential fit
        yhat = np.exp(theta[0])*np.exp(theta[1] * rank[1])
    return yhat

def plotPtsvsRank(pdict,pos=['QB','RB','WR','TE','K','Def'],fitfuns=['e','e','e','e','l','l']):
    corder=['black','red','blue','green','purple','orange']
    fig1=plt.figure()
    allfitcoef={}
    for i,p in enumerate(pos):
        # plot the adp vs points
        plt.plot(range(len(pdict[p][0])),pdict[p][0],marker='o',linestyle='',color=corder[i],alpha=0.2)
        # fit
        fitpredcoefs=fitPtsvRank(pdict[p][0],fitfuns[i])
        fitpred=predictPtsvRank(fitpredcoefs,len(pdict[p][0]),fitfuns[i])
        plt.plot(fitpred,color=corder[i],linestyle='--',\
        label=p+' v ADP', linewidth=2)
        # plot the rank order
        npred = len(pdict[p][2])
        nfit = 10#npred/2
        plt.plot(pdict[p][2],color=corder[i],linewidth=2,label=p+' v Actual')
        fitactcoefs = fitPtsvRank(pdict[p][2][:nfit],fitfuns[i])
        fitact= predictPtsvRank(fitactcoefs,npred,fitfuns[i])
        allfitcoef[p]=[fitpredcoefs,fitactcoefs]
        plt.plot(fitact,color=corder[i],linewidth=0.5,label=p+' v Actual fit')
        
    plt.legend()
    plt.ylabel('Fantasy Points')
    plt.xlabel('Rank order ADP or Actual')
    plt.show()
    return allfitcoef

def getCurrSeasonRanks(src='FantasyPros_2014_Preseason_Overall_Rankings.csv'):
    # get rankings of current season players from average draft pick
    # from Fantasy Pros in csv format
    df=pd.read_csv(src)
    # return dataframe of ranks, stdev of ranks, adp for each player
    return df

def projectPoints(rank,histdata):
    # project points based on position and draft order using historical data
    # histdata is a fit to the rank within the position
    # return projected points
    return histdata[rank]


