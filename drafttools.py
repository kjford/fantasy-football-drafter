'''
Fantasy Football drafting tool
'''
import difflib
import numpy as np
import pandas as pd
import time

# default params
# note that order for fit weights, posexp, denom, and VORper are: ['QB','RB','WR','TE','K','DST']
params={'fitcol':'ADP',\
        'DST': ([ 64.43846154,0],[109.15636364,-7.04363636]),\
        'K': ([137.93333333,-4.81818182],[165.40363636,-8.39636364]),\
        'QB': ([5.71847324,-0.02947322],[5.87262851,-0.03871626]),\
        'RB': ([5.31609401,-0.02356142],[5.65510787,-0.04952052]),\
        'TE': ([4.77645994,-0.02542146],[5.13387644,-0.07972913]),\
        'WR': ([5.13399519,-0.01708781],[5.38410389,-0.02901066]),\
        'fitweight':[0.5,0.5,0.5,1.,1.,1.],\
        'posexp':[2.,5.,5.,1.,1.,1.],\
        'denom':[1.5,2.5,2.5,0.5,50.,50.],\
        'VORper':[25,10,10,10,25,25]}

# default data
data=pd.read_csv('FantasyPros_2014_Preseason_Overall_Rankings.csv')


class Draft:
    def __init__(self,managers,models,masterroster=data.Player.tolist(),pos=data.Position.tolist(),rounds=15,fullauto=False):
        self.managers=managers
        self.models=models
        self.masterroster=masterroster
        self.poslist=pos
        self.n=len(managers)
        self.rounds=rounds
        self.onPick=-1
        self.onRound=-1
        self.picker=-1
        self.rosters={x:None for x in managers}
        self.picked=None
        x=range(self.n)
        self.order=[k for subarr in [x if z%2==0 else x[::-1] for z in range(rounds)] for k in subarr]
        self.fullauto=fullauto
    
    def __repr__(self):
        if self.onPick>-1:
            return 'Draft not started yet'
        else:
            return 'Round#: %i \nPick#: %i \nUp: %s'%(self.onRound+1,self.onPick+1,self.picker)
    
    def pick(self,player):
        # check if picked
        if self.picked:
            if player in self.picked:
                print 'Already chosen, pick again'
                self.query()
                return
        # add to roster
        if self.rosters[self.picker]:
            self.rosters[self.picker].append(player)
        else:
            self.rosters[self.picker]=[player]
        for m in self.models:
            m.update(player)
        # remove from master
        pind=self.masterroster.index(player)
        pos='UNK'
        if pind>-1:
            self.masterroster.pop(pind)
            pos=self.poslist[pind]
            self.poslist.pop(pind)
        # add to picked
        if self.picked:
            self.picked.append((player,pos))
        else:
            self.picked=[(player,pos)]
        # move on to next picker
        self.next()
    
    def next(self):
        # move to next person in draft
        if self.onPick==(len(self.order)-1):
            print 'Draft complete'
            self.saveResults()
        else:
            ctext='''
            (0) Make a selection \n
            (1) Check player roster \n
            (2) Save\n
            Input: '''
            choice=''
            if self.fullauto:
                choice='0'
            while choice not in ['0','1','2']:
                choice = raw_input(ctext)
            choice=int(choice)
            if choice==1:
                self.checkroster()
                self.next()
            if choice==2:
                self.saveResults()
            if choice==0:
                # update
                self.onPick+=1
                self.picker=self.managers[self.order[self.onPick]]
                if self.onPick%self.n==0:
                    self.onRound+=1
                    print 'Starting Round %i'%(self.onRound+1)
                print 'Round#: %i \nPick#: %i \nUp: %s'%(self.onRound+1,self.onPick+1,self.picker)
                self.query() 
    
    def checkroster(self):
        # check a player's roster
        choice = raw_input('Which player?: ')
        if choice in self.managers:
            print self.rosters[choice]
        self.next()
    
    def query(self):
        # ask the current picker to make a selection
        currentmodel = self.models[self.order[self.onPick]]
        if currentmodel.auto:
            print 'Auto selecting: '
            selection=currentmodel.recommend()
            print selection[0] # this is a list of lists
            raw_input('Enter to confirm')
            self.pick(selection[0][0])
        else:
            # no model, choose manually
            if currentmodel.type==None:
                while True:
                    selection= raw_input('Choose a player: ')
                    # check
                    selection = self.checkinput(selection)
                    conf = raw_input('Confirm (y/n)?: ')
                    if conf=='y':
                        break
                self.pick(selection)
            else:
                # get recommendations
                print 'Recommendations:'
                selection=currentmodel.recommend(10)
                # make this prettier
                print 'Choice\tPlayer\t\tPos\tTeam\tADP\tVOR\tSTDEV\tBestRank'
                for i,sel in enumerate(selection):
                    print str(i)+'\t'+'\t'.join([str(x) for x in sel])
                choice=''
                if self.fullauto:
                    choice='0'
                while len(choice)<1:
                    choice = raw_input('Choose a recommendation # or -1 to manually input: ')
                choice=int(choice)
                if choice>-1:
                    print 'You chose %s'%selection[choice]
                    if self.fullauto:
                        self.pick(selection[choice][0])
                    else:
                        raw_input('Enter to confirm')
                        self.pick(selection[choice][0])
                else:
                    while True:
                        selection= raw_input('Choose a player: ')
                        # check
                        selection = self.checkinput(selection)
                        conf = raw_input('Confirm (y/n)?: ')
                        if conf=='y':
                            break
                    self.pick(selection)
    
    def checkinput(self,inputstr):
        # checks to see if the input string is in the master list
        while True:
            closest=difflib.get_close_matches(inputstr,self.masterroster)
            if closest[0]==inputstr:
                break
            print 'Did you mean: '+'or '.join(closest)
            inputstr = raw_input('Re-enter: ')
        return inputstr   
                    
    
    
    def saveResults(self):
        # save out to csv all picks
        ts = time.time()
        ts=str(int(ts))
        with open('draftresults_'+ts+'.csv','w') as f:
            f.write('Player,Position,Manager\n')
            for i,pick in enumerate(self.picked):
                f.write(pick[0]+','+pick[1]+','+self.managers[self.order[i]]+'\n')
        for k,m in enumerate(self.models):
            print self.managers[k]
            print m.roster
    
    def start(self):
        # start the draft
        print 'Starting Draft'
        self.onPick=0
        self.onRound=0
        self.picker=self.managers[0]
        print 'Round#: %i \nPick#: %i \nUp: %s'%(self.onRound+1,self.onPick+1,self.picker)
        self.query()



class Model:
    # model for drafting players
    # data is a pandas dataframe with Rank, Player, Position, Team, Bye, BestRank, WorstRank,
    # AvgRank, Std, and ADP
    # computes VOR given some input parameters in params
    # -fit coefficients of points for position rank actual and predicted
    # -weighting of fit functions 0 use only predicted fit to 1 for use actual fit
    # -expected number at QB,RB,WR,TE,K,Def positions (default 2,5,5,1,1,1)
    # -denominator for filling weight (default is 1.5,2.5,2.5,0.5,2,2, this lowers K and DST)
    # note: lower this to increase bias toward high positions with more spots
    
    
    def __init__(self,type='VOR',auto=False,data=data,draftpos=0,nplayers=10,rounds=15,params=params):
        self.type=type
        self.auto=auto
        self.data=data
        self.draftpos=draftpos
        self.nplayers=nplayers
        self.rounds=rounds
        self.state=0
        x=range(nplayers)
        self.order=[k for subarr in [x if z%2==0 else x[::-1] for z in range(rounds)] for k in subarr]
        self.roster=None
        self.params=params
        self.computeVOR()
    
    
    def __repr__(self):
        if self.roster:
            return self.roster
        else:
            return 'None'
    
    def recommend(self,n=5):
        # recommend players to draft based on who is left ordered by value
        recos=self.data.sort('VOR',ascending=0)
        recos=recos.iloc[:n]
        return np.array(recos[['Player','Position','Team','ADP','VOR','Std','BestRank']])
    
    def update(self,player):
        # find the player in the data, remove, and update the model
        if self.data:
            # find player in data
            playerdata=self.data[self.data['Player']==player]
            # if the state of the model is one of my selections, add to my roster
            if self.order[self.state]==self.draftpos:
                if self.roster:
                    self.roster=pd.concat([self.roster,playerdata])
                else:
                    self.roster=pd.DataFrame(playerdata)
            self.data=self.data[self.data['Player']!=player]
            # update the model, change VOR if necessary
            self.updateVOR()
        self.state+=1
    
    def updateVOR(self):
        # update the VOR based on draft position
        # get count of each position in roster
        pos=['QB','RB','WR','TE','K','DST']
        for i,p in enumerate(pos):
            if self.roster:
                nfilled=(self.roster['Position']==p).sum()
            else:
                nfilled=0.0
            nexp = self.params['posexp'][i]
            openslots = nexp - nfilled
            weight = openslots*1.0/(self.params['denom'][i])
            if self.type=='VOR':
                # scale VOR
                self.data['VOR'][self.data['Position']==p] = self.data['VORraw'][self.data['Position']==p] * weight
        
        
        
    def computeVOR(self):
        # take the model input and compute VOR for each player
        # simple model is just the draft position
        if self.type=='ADP':
            self.data['VOR'] = 1.0/self.data['ADP']
            self.data=self.data[self.data['ADP'].isnull()<1]
        elif self.type=='Rank':
            self.data['VOR'] = 1.0/self.data['Rank']
        else: # actually compute VOR
            # group by position sorted on ADP
            self.data['VORraw']=0.0
            self.data=self.data[self.data[self.params['fitcol']].isnull()<1]
            for gr,subset in self.data.groupby('Position'): 
                # order
                orderind=['QB','RB','WR','TE','K','DST'].index(gr)
                fitcol=self.params['fitcol']
                ordered=subset.sort(fitcol)
                l = len(ordered)
                ranks=np.arange(l)*1.0
                # get fits (exponential)
                if gr in ['QB','RB','WR','TE']:
                    fitprep = np.exp(self.params[gr][0][0]) * np.exp(self.params[gr][0][1]*ranks)
                    fitact = np.exp(self.params[gr][1][0]) * np.exp(self.params[gr][1][1]*ranks)    
                else: # linear for K and Def
                    fitprep = self.params[gr][0][0] + self.params[gr][0][1]*ranks
                    fitact = self.params[gr][1][0] + self.params[gr][1][1]*ranks
                
                fit = fitact*self.params['fitweight'][orderind] + fitprep*(1.0-self.params['fitweight'][orderind])
                replacement=np.percentile(fit,self.params['VORper'][orderind])
                r=0
                for ind,row in ordered.iterrows():
                    self.data['VORraw'].loc[ind] = fit[r] - replacement
                    r+=1
            self.data['VOR']=self.data['VORraw']
            self.updateVOR()

        