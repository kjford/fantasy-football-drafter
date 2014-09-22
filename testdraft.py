'''
Run Fantasy football draft
'''
import drafttools

# who is playing
managers = ['Player1','Player2','Player3','Player4','Player5','Player6',\
'Player7','Player8','Player9','Player10']

# this will be different
models=[drafttools.Model(type='VOR',draftpos=0)]
for i in xrange(9):
    models.append(drafttools.Model(type='VOR',draftpos=i+1))

# init draft
thedraft = drafttools.Draft(managers,models,fullauto=True)

thedraft.start()