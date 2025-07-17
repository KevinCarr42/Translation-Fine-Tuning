import random
import json
from translate import translate_text


SEP = "<sep>"

def sample_jsonl(path, n=100):
    with open(path, 'r') as f:
        lines = random.sample(list(f), n)
    return [json.loads(line) for line in lines]


sample_translations = sample_jsonl("training_data.jsonl", 10)

for d in sample_translations:
    full_text = d['text']

    lang = full_text.split(' ')[1]
    lang = {"English": "en", "French": "fr"}[lang]

    both_lang = full_text.split(":")
    if len(both_lang) != 2:
        print(f'too many colons!  ->  {full_text}')
        continue

    split_lang = both_lang[1].split(SEP)
    if len(split_lang) != 2:
        print(f'too many separators!  ->  {split_lang}')
        continue
    text_in, text_out = split_lang
    text_in, text_out = text_in.strip(), text_out.strip()

    translated_text_base = translate_text(text_in, lang, False)
    translated_text_finetuned = translate_text(text_in, lang, True)

    print()
    print('\ttext in')
    print(text_in)
    print('\ttext out (expected)')
    print(text_out)
    print('\ttext out (predicted with base model)')
    print(translated_text_base)
    print('\ttext out (predicted with finetuned model)')
    print(translated_text_finetuned)
    print()


"""
how well did it do? bad. but the base model is also bad. need better prompting


	text in
La pêche commerciale couvre en grande partie l ensemble de la zone
	text out (expected)
The commercial fishery covers most of the area
	text out (predicted with base model)
La pêche commerciale couvre en grande partie l ensemble de la zone économique exclusive (ZEE) du Sénégal, soit 200 milles nautiques (environ 370 km) autour des côtes sénégalaises. La pêche artisanale, quant à elle, est pratiquée dans les eaux territoriales (12 milles nautiques, soit environ 22 km autour des côtes sénégalaises). La pêche artisanale sénégalaise emploie environ 600 000 personnes, soit 10 % de la population
	text out (predicted with finetuned model)
La pêche commerciale couvre en grande partie l ensemble de la zone of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing covers most of the area of interest (AOI) (Figure 2) Commercial fishing


	text in
Eastern Shore Islands study area (blue shaded and outlined area) with locations referenced within this document
	text out (expected)
Zone d étude des îles de la côte Est (zone en bleu et délimitée) avec les emplacements mentionnés dans le document........................................................................
	text out (predicted with base model)
Eastern Shore Islands study area (blue shaded and outlined area) with locations referenced within this document (red dots) (Figure 1) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombragée et délimitée) avec les emplacements référencés dans le présent document (points rouges) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombragée et délimitée) avec les emplacements référencés dans le présent document (points rouges) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombr
	text out (predicted with finetuned model)
Eastern Shore Islands study area (blue shaded and outlined area) with locations referenced within this document (red dots) (Figure 1) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombragée et délimitée) avec les emplacements référencés dans le présent document (points rouges) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombragée et délimitée) avec les emplacements référencés dans le présent document (points rouges) Figure 1 : Zone d étude des îles de la côte Est (zone bleue ombr


	text in
V9T 1K3 1 This series documents the scientific basis for the evaluation of fisheries resources in Canada
	text out (expected)
1 La présente série documente les bases scientifiques des évaluations des ressources halieutiques du Canada
	text out (predicted with base model)
V9T 1K3 1 This series documents the scientific basis for the evaluation of fisheries resources in Canada and is published by the Department of Fisheries and Oceans Canada (DFO) at the request of the Deputy Minister of Fisheries and Oceans Canada (DFO) and the Regional Directors-General of the Department of Fisheries and Oceans Canada (DFO) and the Canadian Food Inspection Agency (CFIA) as a means of informing the public about the status of the resource and the measures being taken for its conservation and sustainable use 1 La présente série documente les bases scientifiques des évaluations des ressources halieutiques du Canada et est publiée à la
	text out (predicted with finetuned model)
V9T 1K3 1 This series documents the scientific basis for the evaluation of fisheries resources in Canada and is published by the Department of Fisheries and Oceans Canada (DFO) at the request of the Deputy Minister of Fisheries and Oceans Canada (DFO) and the Regional Directors-General of the Department of Fisheries and Oceans Canada (DFO) and the Canadian Food Inspection Agency (CFIA) as a means of informing the public about the status of the resource and the measures being taken for its conservation and sustainable use 1 La présente série documente les bases scientifiques des évaluations des ressources halieutiques du Canada et est publiée à la


	text in
Toutes les concentrations modélisées (5 m supérieurs) sont inférieures à 103 UFP m3
	text out (expected)
All modelled concentrations (upper 5 m) are under 103pfu m3
	text out (predicted with base model)
Toutes les concentrations modélisées (5 m supérieurs) sont inférieures à 103 UFP m3 (all modeled concentrations (5 m above) are less than 103 UFP m3) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2)
	text out (predicted with finetuned model)
Toutes les concentrations modélisées (5 m supérieurs) sont inférieures à 103 UFP m3 (all modeled concentrations (5 m above) are less than 103 UFP m3) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2) (Figure 2)


	text in
Nous présentons également l estimation acoustique le long des traces laissées par le relevé
	text out (expected)
We also present acoustic estimation along the survey tracks
	text out (predicted with base model)
Nous présentons également l estimation acoustique le long des traces laissées par le relevé (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the ac
	text out (predicted with finetuned model)
Nous présentons également l estimation acoustique le long des traces laissées par le relevé (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the acoustic estimate along the tracks of the survey (transects) and the ac


	text in
Tendances relatives à l estimation de la biomasse (âges 2 à 8) et de la BSR à l aide d une analyse de cohorte fondée sur les données du relevé du MPO
	text out (expected)
Trends in biomass (ages 2-8) and SSB estimated from cohort analysis of DFO survey data
	text out (predicted with base model)
Tendances relatives à l estimation de la biomasse (âges 2 à 8) et de la BSR à l aide d une analyse de cohorte fondée sur les données du relevé du MPO (DFO) (2000-2018) (top) and trends in biomass estimates (ages 2 to 8) and SSB using a cohort analysis based on DFO survey data (2000-2018) (bottom) (top) and trends in biomass estimates (ages 2 to 8) and SSB using a cohort analysis based on DFO survey data (2000-2018) (bottom) (top) and trends in biomass estimates (ages 2 to 8) and
	text out (predicted with finetuned model)
Tendances relatives à l estimation de la biomasse (âges 2 à 8) et de la BSR à l aide d une analyse de cohorte fondée sur les données du relevé du MPO (DFO) (2000-2018) (top) and trends in biomass estimates (ages 2 to 8) and SSB using a cohort analysis based on DFO survey data (2000-2018) (bottom) (top) and trends in biomass estimates (ages 2 to 8) and SSB using a cohort analysis based on DFO survey data (2000-2018) (bottom) (top) and trends in biomass estimates (ages 2 to 8) and



"""