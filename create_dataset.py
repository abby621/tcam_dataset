from core.models import *
import numpy as np
import random

app_hotels = np.unique(list(Image.objects.filter(capture_method__name__icontains='app').values_list('hotel_id',flat=True)))

ims = {}
for hotel in app_hotels:
    tc_ims = Image.objects.filter(capture_method__name__icontains='app',hotel=hotel)
    ex_ims = Image.objects.filter(hotel=hotel).exclude(capture_method__name__icontains='app')
    if tc_ims.count() > 0 and ex_ims.count() > 4:
        ims[hotel] = {}
        ims[hotel]['ims'] = {}
        ims[hotel]['tcam'] = []
        ims[hotel]['expedia'] = []
        tmstps = np.array(list(tc_ims.values_list('upload_timestamp',flat=True)))
        unique_times = np.unique(tmstps)
        ims_by_user = {}
        for t in range(0,unique_times.shape[0]):
            inds = np.where(tmstps==unique_times[t])[0]
            ims_by_user[t] = [(i.id,i.path,i.upload_timestamp) for i in np.array(tc_ims)[inds]]
        ims[hotel]['numUsers'] = len(ims_by_user.keys())
        ims[hotel]['ims']['tcam'] = ims_by_user
        ims[hotel]['ims']['expedia'] = list(ex_ims.values_list('id','path','upload_timestamp'))
        h = Hotel.objects.get(id=hotel)
        ims[hotel]['id'] = h.id
        ims[hotel]['name'] = h.name
        ims[hotel]['lat'] = h.lat
        ims[hotel]['lng'] = h.lng
        print hotel

queryHotels = random.sample(ims.keys(),6000)
train_set = {}
test_set = {}

for hotel in ims.keys():
    print hotel
    if not hotel in train_set.keys():
        train_set[hotel] = {}
        train_set[hotel]['ims'] = {}
        train_set[hotel]['ims']['tcam'] = {}
        train_set[hotel]['ims']['expedia'] = {}
        train_set[hotel]['id'] = ims[hotel]['id']
        train_set[hotel]['name'] = ims[hotel]['name']
        train_set[hotel]['lat'] = ims[hotel]['lat']
        train_set[hotel]['lng'] = ims[hotel]['lng']
    if hotel in queryHotels:
        if not hotel in test_set.keys():
            test_set[hotel] = {}
            test_set[hotel]['ims'] = {}
            test_set[hotel]['ims']['tcam'] = {}
            test_set[hotel]['id'] = ims[hotel]['id']
            test_set[hotel]['name'] = ims[hotel]['name']
            test_set[hotel]['lat'] = ims[hotel]['lat']
            test_set[hotel]['lng'] = ims[hotel]['lng']
        userInd = random.choice(ims[hotel]['ims']['tcam'].keys())
        queryIm = random.choice(ims[hotel]['ims']['tcam'][userInd])
        test_set[hotel]['ims']['tcam'][queryIm[0]] = {'path':queryIm[1],'time':queryIm[2].strftime("%Y-%m-%d %H:%M:%S")}
        for user in ims[hotel]['ims']['tcam'].keys():
            if user != userInd:
                numToInclude = random.choice(range(len(ims[hotel]['ims']['tcam'][user])))
                whichToInclude = random.sample(ims[hotel]['ims']['tcam'][user],numToInclude)
                for im in whichToInclude:
                    train_set[hotel]['ims']['tcam'][im[0]] = {'path':im[1],'time':im[2].strftime("%Y-%m-%d %H:%M:%S")}
    else:
        for user in ims[hotel]['ims']['tcam'].keys():
            numToInclude = random.choice(range(len(ims[hotel]['ims']['tcam'][user])))
            whichToInclude = random.sample(ims[hotel]['ims']['tcam'][user],numToInclude)
            for im in whichToInclude:
                train_set[hotel]['ims']['tcam'][im[0]] = {'path':im[1],'time':im[2].strftime("%Y-%m-%d %H:%M:%S")}
    for im in ims[hotel]['ims']['expedia']:
        train_set[hotel]['ims']['expedia'][im[0]] = {'path':im[1],'time':''}

allHotels = list(Hotel.objects.exclude(id__in=train_set.keys()).exclude(id__in=test_set.keys()).values_list('id',flat=True))
ctr = 0
while len(train_set.keys()) < 60000:
    hotel = int(allHotels[ctr])
    ex_ims = list(Image.objects.filter(hotel=hotel).exclude(capture_method__name__icontains='app').values_list('id','path','upload_timestamp'))
    if len(ex_ims) > 4:
        train_set[hotel] = {}
        train_set[hotel]['ims'] = {}
        train_set[hotel]['ims']['tcam'] = {}
        train_set[hotel]['ims']['expedia'] = {}
        h = Hotel.objects.get(id=hotel)
        train_set[hotel]['id'] = h.id
        train_set[hotel]['name'] = h.name
        train_set[hotel]['lat'] = h.lat
        train_set[hotel]['lng'] = h.lng
        for im in ex_ims:
            train_set[hotel]['ims']['expedia'][im[0]] = {'path':im[1],'time':''}
        print len(train_set.keys())
    ctr += 1

numImsTotal = 0
numExpedia = 0
numTcam = 0
for h in train_set.keys():
    if 'expedia' in train_set[h]['ims'].keys():
        numImsTotal += len(train_set[h]['ims']['expedia'].keys())
        numExpedia += len(train_set[h]['ims']['expedia'].keys())
    if 'tcam' in train_set[h]['ims'].keys():
        numImsTotal += len(train_set[h]['ims']['tcam'].keys())
        numTcam += len(train_set[h]['ims']['tcam'].keys())

chainNames=["Best Western","Hyatt","Marriott","Hilton","Adagio","Gaylord","Nikko","Rosewood","Grand Mercure","Kempinski","Pan Pacific","Curio","Langham","Protea","Aman","St. Regis","Lotte","Conrad","Oaks","Cambria Suites","Magnuson","Mainstay Suites","GuestHouse","Prince","Millennium","W Hotel","Sofitel","Best Western Premier","Pullman","Coast","Loews","Delta","Grand Hyatt","Le Meridien","Park Inn","Fairmont","Four Seasons","Knights Inn","Ritz-Carlton","Scandic","Hawthorn Suites","Omni","Red Lion","AmericInn","Ascend","ibis","Mercure","Kimpton","Home2 Suites","Aloft","InterContinental","Novotel","Baymont Inn & Suites","Wingate","Howard Johnson","Four Points","Rodeway Inn","Travelodge","Clarion","Renaissance","Wyndham","Staybridge Suites","Americas Best Value Inn","Microtel","Candlewood Suites","Radisson","Westin","Red Roof","TownePlace Suites","Sleep Inn","Econo Lodge","Crowne Plaza","Extended Stay America","Ramada","Embassy Suites","Motel 6","Sheraton","Homewood Suites","SpringHill Suites","Comfort Suites","DoubleTree","Residence Inn","Best Western Plus","Super 8","Quality Inn","Days Inn","La Quinta","Hilton Garden Inn","Comfort Inn","Fairfield Inn","Courtyard by Marriott","Hampton","Holiday Inn"]

for h in train_set.keys():
    cnName = ''
    cnId = -1
    for ci in range(len(chainNames)):
        cn = chainNames[ci]
        if cn.lower() in train_set[h]['name'].lower():
            cnName = cn
            cnId = ci
    train_set[h]['chainName'] = cnName
    train_set[h]['chainId'] = cnId

for h in test_set.keys():
    cnName = ''
    cnId = -1
    for ci in range(len(chainNames)):
        cn = chainNames[ci]
        if cn.lower() in test_set[h]['name'].lower():
            cnName = cn
            cnId = ci
    test_set[h]['chainName'] = cnName
    test_set[h]['chainId'] = cnId

import json

with open('/home/abby/test_set.json','w') as test_file:
    json.dump(test_set,test_file)

with open('/home/abby/train_set.json','w') as train_file:
    json.dump(train_set,train_file)
