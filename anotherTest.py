import os
import json
arr = [
    'ROOT0',
    'ROOT1',
    'ROOT2',
    'ROOT3',
    'ROOT4',
    'ROOT5',
    'ROOT7',
    'ROOT8',
    'ROOT9',
    'ROOT10',
    'ROOT11',
    'ROOT12',
    'ROOT13',
    'ROOT14',
    'ROOT15',
    'ROOT16',
    'ROOT17',
    'ROOT20',
    'ROOT21',
    'ROOT22',
    'ROOT23',
    'ROOT24',
    'ROOT25',
    'ROOT26',
    'ROOT27',
    'ROOT28',
    'ROOT29',
    'ROOT30',
    'ROOT31',
    'ROOT32',
    'ROOT33',
    'ROOT35',
    'ROOT36',
    'ROOT37',
    'ROOT38',
    'ROOT41',
    'ROOT42',
    'ROOT43',
    'ROOT44',
    'ROOT46',
    'ROOT47',
    'ROOT49',
    'ROOT50',
    'ROOT51',
    'ROOT52',
    'ROOT53',
    'ROOT54',
    'ROOT55',
    'ROOT56',
    'ROOT57',
    'ROOT58',
    'ROOT59',
    'ROOT60',
    'ROOT61',
    'ROOT62',
    'ROOT64',
    'ROOT65',
    'ROOT66',
    'ROOT67',
    'ROOT68',
    'ROOT70',
    'ROOT71',
    'ROOT72',
    'ROOT73',
    'ROOT74',
    'ROOT75',
    'ROOT76',
    'ROOT77',
    'ROOT78',
    'ROOT79',
    'ROOT80',
    'ROOT81',
    'ROOT82',
    'ROOT83',
    'ROOT84',
    'ROOT85',
    'ROOT86',
    'ROOT88',
    'ROOT89',
    'ROOT92',
    'ROOT93',
    'ROOT94',
    'ROOT95',
    'ROOT96',
    'ROOT98',
    'ROOT99',
    'ROOT100',
    'ROOT101',
    'ROOT102',
    'ROOT103',
    'ROOT104',
    'ROOT106',
    'ROOT107',
    'ROOT109',
    'ROOT110',
    'ROOT112',
    'ROOT113',
    'ROOT114',
    'ROOT116',
    'ROOT118',
    'ROOT120',
    'ROOT121',
    'ROOT122',
    'ROOT124',
    'ROOT125',
    'ROOT126',
    'ROOT128',
    'ROOT129',
    'ROOT130',
    'ROOT131',
    'ROOT132',
    'ROOT134',
    'ROOT135',
    'ROOT136',
    'ROOT137',
    'ROOT139',
    'ROOT140',
    'ROOT141',
    'ROOT142',
    'ROOT143',
    'ROOT144',
    'ROOT145',
    'ROOT147',
    'ROOT149',
    'ROOT150',
    'ROOT151',
    'ROOT152',
    'ROOT153',
    'ROOT154',
    'ROOT155',
    'ROOT157',
    'ROOT159',
    'ROOT160',
    'ROOT161',
    'ROOT162',
    'ROOT163',
    'ROOT164',
    'ROOT166',
    'ROOT167',
    'ROOT168',
    'ROOT169',
    'ROOT170',
    'ROOT171',
    'ROOT173',
    'ROOT174',
    'ROOT175',
    'ROOT176',
    'ROOT178',
    'ROOT179',
    'ROOT180',
    'ROOT181',
    'ROOT182',
    'ROOT183',
    'ROOT185',
    'ROOT186',
    'ROOT187',
    'ROOT189',
    'ROOT190',
    'ROOT191',
    'ROOT192',
    'ROOT193',
    'ROOT194',
    'ROOT195',
    'ROOT196',
    'ROOT199',
    'ROOT200',
    'ROOT201',
    'ROOT202',
    'ROOT203',
    'ROOT205',
    'ROOT206',
    'ROOT207',
    'ROOT208',
    'ROOT209',
    'ROOT210',
    'ROOT211',
    'ROOT212',
    'ROOT214',
    'ROOT215',
    'ROOT216',
    'ROOT217',
    'ROOT218',
    'ROOT219',
    'ROOT220',
    'ROOT222',
    'ROOT223',
    'ROOT224',
    'ROOT225',
    'ROOT228',
    'ROOT229',
    'ROOT230',
    'ROOT231',
    'ROOT232',
    'ROOT233',
    'ROOT235',
    'ROOT236',
    'ROOT237',
    'ROOT238',
    'ROOT239',
    'ROOT240',
    'ROOT241',
    'ROOT242',
    'ROOT243',
    'ROOT244',
    'ROOT246',
    'ROOT247',
    'ROOT248',
    'ROOT249',
    'ROOT250',
    'ROOT251',
    'ROOT252',
    'ROOT253',
    'ROOT254',
    'ROOT257',
    'ROOT258',
    'ROOT259',
    'ROOT261',
    'ROOT262',
    'ROOT263',
    'ROOT264',
    'ROOT266',
    'ROOT267',
    'ROOT269',
    'ROOT270',
    'ROOT272',
    'ROOT273',
    'ROOT275',
    'ROOT276',
    'ROOT277',
    'ROOT278',
    'ROOT280',
    'ROOT282',
    'ROOT283',
    'ROOT284',
    'ROOT285',
    'ROOT286',
    'ROOT287',
    'ROOT288',
    'ROOT291',
    'ROOT292',
    'ROOT294',
    'ROOT295',
    'ROOT297',
    'ROOT298',
    'ROOT299',
    'ROOT300',
    'ROOT301',
    'ROOT302',
    'ROOT303',
    'ROOT304',
    'ROOT305',
    'ROOT306',
    'ROOT308',
    'ROOT309',
    'ROOT310',
    'ROOT312',
    'ROOT313',
    'ROOT314',
    'ROOT315',
    'ROOT316',
    'ROOT318',
    'ROOT319',
    'ROOT320',
    'ROOT321',
    'ROOT322',
    'ROOT323',
    'ROOT324',
    'ROOT325',
    'ROOT326',
    'ROOT328',
    'ROOT331',
    'ROOT333',
    'ROOT336',
    'ROOT337',
    'ROOT338',
    'ROOT339',
    'ROOT340',
    'ROOT341',
    'ROOT342',
    'ROOT343',
    'ROOT344',
    'ROOT345',
    'ROOT346',
    'ROOT348',
    'ROOT350',
    'ROOT351',
]


arr2 = []

with open('SmallData/utterances.jsonl', 'r') as fr:
    for line in fr:
        line = json.loads(line)
        if line['conversation_id'] in arr:
            arr2.append(line)

if os.path.exists('SmallData/utterances.jsonl'):
    os.remove('SmallData/utterances.jsonl')

with open('SmallData/utterances.jsonl', 'a') as fw:
    for line in arr2:
        fw.write(json.dumps(line) + '\n')
