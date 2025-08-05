import random
import json
from translate import translate_text


def sample_jsonl(path, n=100):
    with open(path, 'r') as f:
        lines = random.sample(list(f), n)
    return [json.loads(line) for line in lines]


sample_translations = sample_jsonl("training_data.jsonl", 10)

for d in sample_translations:
    source = d.get('source')
    target = d.get('target')
    source_lang = d.get('source_lang')
    translated_base_model = translate_text(source, source_lang, False)
    translated_finetuned = translate_text(source, source_lang, True)

    print()
    print(source_lang, '\ttext in')
    print('\t', source)
    print('text out (expected)')
    print('\t', target)
    print('text out (predicted with base model)')
    print('\t', translated_base_model)
    print('text out (predicted with finetuned model)')
    print('\t', translated_finetuned)
    print()



# TODO: why is this still bad?
"""
/home/carrk/.venv/bin/python /home/carrk/finetune/evaluate.py 
Loading checkpoint shards: 100%|██████████| 19/19 [12:46<00:00, 40.34s/it]
/home/carrk/.venv/lib/python3.10/site-packages/bitsandbytes/nn/modules.py:457: UserWarning: Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.
  warnings.warn(

en 	text in
	 Hydrogen peroxide is fully miscible in water and will remain in the aqueous phase upon entering the environment
text out (expected)
	 Le peroxyde d hydrogène est entièrement miscible dans l eau et demeure en phase aqueuse lorsqu il entre dans l environnement
text out (predicted with base model)
	 L'peroxyde d'hydrogène est entièrement soluble dans l'eau et restera dans la phase aqueuse en pénétrant dansl'environnement
text out (predicted with finetuned model)
	 Le peroxyde d hydrogène est complètement soluble dans l eau et demeurera dans la phase aqueuse lorsqu il pénètre dans l environnement après son utilisation dans le traitement des eaux de ballast des navires commerciaux au Canada atlantique et au Québec 12 l ecosystème marin canadien (p


fr 	text in
	 Il convient de noter que ces menaces ne sont que les résultats directs des nouvelles empreintes des activités de logement et de développement
text out (expected)
	 Note that these threats are only the direct results from new footprints of housing and development activities
text out (predicted with base model)
	 It should be noted that these threats are only the direct results of the new footprints of housing and development activities, and do not include the indirect effects of these activities (e
text out (predicted with finetuned model)
	 It should be noted that these threats are only the direct results of the new footprints of housing and development activities, and do not include the indirect effects of these activities (e


en 	text in
	 The contribution of the second equalisation to minimizing loss of genetic diversity is unclear
text out (expected)
	 La contribution de la deuxième égalisation à la limitation de la perte de diversité génétique est plus ambigüe
text out (predicted with base model)
	 La contribution de la deuxième égalisation à la minimisation de la perte de diversité génétique est incertaine
text out (predicted with finetuned model)
	 La contribution de la deuxième égalisation à la minimisation de la perte de diversité génétique est incertaine


fr 	text in
	 Il convient d effectuer d autres recherches pour quantifier ces relations
text out (expected)
	 More research is required to quantify these relationships
text out (predicted with base model)
	 Further research is needed to quantify these relationships more fully and to determine the extent to which they apply to other species of fish and invertebrates in the area, as well as to other areas of the Gulf of Maine and the Bay of Fundy (BoF) and Scotian Shelf (SS) regions of the Northwest Atlantic Ocean (NWAO) (e
text out (predicted with finetuned model)
	 Further research is needed to quantify these relationships more fully and to determine the extent to which they apply to other species of fish and invertebrates in the area, as well as to other areas of the Gulf of Maine and the Bay of Fundy (BoF) and Scotian Shelf (SS) regions of the Northwest Atlantic Ocean (NWAO) (e


fr 	text in
	 On discute de la prestation d'avis concernant les seuils de productivité en vue d'assurer l'efficacité du modèle
text out (expected)
	 There was a discussion on providing advice on productivity thresholds for model effectiveness
text out (predicted with base model)
	 There was discussion on the provision of advice on productivity thresholds to ensure model efficacy
text out (predicted with finetuned model)
	 There was discussion on the provision of advice on productivity thresholds to ensure model efficacy


en 	text in
	 Unfortunately, such changes reveal little about population structure
text out (expected)
	 Malheureusement, de tels changements ne donnent que peu d'indications sur la structure de la population
text out (predicted with base model)
	 Malheureusement, de tels changements ne révèlent que peu de choses sur la structure de la population concernée
text out (predicted with finetuned model)
	 Malheureusement, de tels changements ne révèlent que peu de choses sur la structure de la population concernée


en 	text in
	 A fishery decision-making framework incorporating the precautionary approach
text out (expected)
	 Un cadre décisionnel pour les pêches intégrant l approche de précaution
text out (predicted with base model)
	 Un cadre décisionnel pour les pêches intégrant l approche de précaution Direction des sciences, Région du Golfe Pêches et Océans Canada Institut Maurice-Lamontagne C
text out (predicted with finetuned model)
	 Un cadre décisionnel pour les pêches intégrant l approche de précaution Direction des sciences, Région du Golfe Pêches et Océans Canada Institut Maurice-Lamontagne C


fr 	text in
	 Il doit reposer sur la littérature ou sur des données concernant l utilisation de l habitat ou la productivité du poisson, idéalement dans la zone localisée
text out (expected)
	 The QAF must be supported by literature or data on habitat use or fish productivity; ideally in the localized area
text out (predicted with base model)
	 It should be based on the literature or data on habitat use or fish productivity, ideally in the localized area of interest (AOI) if available, or in similar areas if not available in the AOI itself (e
text out (predicted with finetuned model)
	 It should be based on the literature or data on habitat use or fish productivity, ideally in the localized area of interest (AOI) if available, or in similar areas if not available in the AOI itself (e


fr 	text in
	 Il faudrait continuer à effectuer régulièrement des relevés indépendants de la pêche dans les SGPP 12 et 19 afin d obtenir une série chronologique indépendante de la pêche pour les estimations de la densité et de surveiller les tendances des populations d oursins verts
text out (expected)
	 The PFMA 12 and PFMA 19 fishery-independent surveys should be continued on a regular basis to provide a fishery independent time-series of density estimates for monitoring Green Sea Urchin population trends
text out (predicted with base model)
	 Regular fishery-independent surveys should continue to be conducted in SGFFs12 and19 to obtain a fishery independent time series for density estimates and to monitor trends in Green Sea Urchin populations within these areas of interest (AOIs) (i
text out (predicted with finetuned model)
	 Regular fishery-independent surveys should continue to be conducted in SGFFs12 and19 to obtain a fishery independent time series for density estimates and to monitor trends in Green Sea Urchin populations within these areas of interest (AOIs) (i


fr 	text in
	 Ainsi, on reconnaît que le fait de travailler dans l Arctique amène un certain nombre d enjeux, y compris des coûts élevés, des conditions souvent hostiles et des difficultés d ordre logistique qui compliquent les activités de recherche et de surveillance
text out (expected)
	 For example, it is recognized that working in the Arctic poses a number of challenges including high costs, often harsh conditions and logistical difficulties that constrain research and monitoring practices
text out (predicted with base model)
	 Thus, it is recognized that working in the Arctic brings with it a number of challenges, including high costs, often hostile conditions, and logistical difficulties that complicate research and monitoring activities in this region of Canada s EEZ and continental shelf areas (CSAs) (DFO 2016a) 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 30
text out (predicted with finetuned model)
	 Thus, it is recognized that working in the Arctic brings with it a number of challenges, including high costs, often hostile conditions, and logistical difficulties that complicate research and monitoring activities in this region of Canada s EEZ and continental shelf areas (CSAs) (DFO 2016a) 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 30


Process finished with exit code 0
"""