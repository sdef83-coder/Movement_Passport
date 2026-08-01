# Movement Passport

Movement Passport est un projet d'analyse vidéo du mouvement destiné à aider les coachs et les cliniciens à objectiver leurs observations, suivre l'évolution d'une personne et adapter un programme d'entraînement ou de rééducation.

Le projet est actuellement un prototype Python utilisant une webcam. Cette interface sert à développer et valider les mesures avant leur intégration dans une future application mobile.

> **Important :** Movement Passport fournit des mesures descriptives et une aide à l'observation. Il ne pose pas de diagnostic médical et ne remplace ni l'examen clinique ni le jugement du professionnel.

## État du projet

- **Module 1 — Squat, vue frontale :** première version publiée avec le tag `v.1.0.0`.
- **Module 2 — Squat, vue sagittale :** première version stable publiée avec le tag `v1.1.0`.

Le module sagittal analyse le côté gauche ou droit présenté à la caméra. Il produit les signaux continus, détecte les répétitions, calcule des indicateurs par répétition et construit un cycle moyen.

Les coordonnées normalisées de MediaPipe sont converties en coordonnées pixels
avant les calculs géométriques. Les angles restent ainsi cohérents entre une
vidéo verticale de téléphone et une image horizontale de webcam.

## Mesures du module sagittal

Les indicateurs principaux sont :

- la flexion du genou ;
- la flexion de hanche ;
- la flexion du tronc ;
- la vitesse angulaire continue du genou ;
- les vitesses moyennes de descente et de remontée ;
- la durée et l'amplitude de chaque répétition ;
- la moyenne et la variabilité des cycles.

La dorsiflexion de cheville et l'inclinaison du pied sont conservées comme mesures secondaires exploratoires. Elles dépendent fortement de la visibilité du talon et de l'avant-pied et doivent donc être interprétées avec prudence.

## Installation

L'environnement testé utilise Python 3.11.

```bash
conda create -n movement-passport python=3.11
conda activate movement-passport
python -m pip install -r requirements.txt
```

Dans Spyder, sélectionnez l'interpréteur Python de l'environnement `movement-passport`, puis placez le dossier de travail à la racine du projet.

## Lancer le module sagittal dans Spyder

Ouvrez `scripts/squat_side_view.py` et exécutez le fichier avec **F5**.

1. Dans la console Spyder, tapez `G` ou `D` selon le côté présenté à la caméra, puis appuyez sur **Entrée**.
2. Cliquez dans la fenêtre de la webcam afin qu'elle reçoive les commandes clavier.
3. Appuyez sur la touche `s` dans la fenêtre de la webcam, sans appuyer sur Entrée.
4. Un décompte de 10 secondes est suivi d'une baseline de 3 secondes en position debout et immobile.
5. Réalisez les squats. Levez une main pour terminer l'enregistrement, ou appuyez sur `q`.

Si la baseline est invalide, le programme revient en attente et indique le marqueur ou le signal limitant. Corrigez le cadrage, puis recommencez avec `s`.

Le script peut aussi être lancé depuis un terminal ouvert à la racine du projet :

```bash
python scripts/squat_side_view.py
```

## Analyser une vidéo sagittale enregistrée

Le module sagittal peut également analyser une vidéo déjà enregistrée avec un téléphone ou une autre caméra. Les calculs, la segmentation et les rapports sont identiques à ceux du mode webcam.

Dans Spyder, ouvrez `scripts/squat_side_video.py` et exécutez le fichier avec **F5**.

1. Choisissez la vidéo dans la fenêtre de sélection de fichiers.
2. Dans la console Spyder, tapez `G` ou `D` selon le côté du corps visible, puis appuyez sur **Entrée**.
3. Indiquez à quelle seconde commence la posture debout immobile. Appuyez directement sur **Entrée** si elle commence dès le début de la vidéo.
4. Le programme utilise les 3 secondes suivantes comme baseline et analyse ensuite tout le reste de la vidéo.
5. Consultez les fichiers générés dans le nouveau sous-dossier de `results/`.

Une vidéo `sagittal_annotated.mp4` est aussi enregistrée dans le dossier de
résultats. Elle montre le squelette détecté, les six marqueurs réellement
utilisés du côté choisi, les angles calculés et la qualité minimale de
détection. Le marqueur le moins visible est signalé en rouge. Cette vidéo sert
au contrôle du suivi et ne modifie pas les calculs.

Formats acceptés : MP4, MOV, M4V, AVI et MKV.

```bash
python scripts/squat_side_video.py
```

Une vidéo sans 3 secondes de posture debout exploitable est volontairement refusée. Cette sécurité évite de produire des angles relatifs et des indicateurs de vitesse à partir d'une mauvaise référence.

## Analyser une vidéo frontale enregistrée

Le module frontal accepte maintenant les vidéos filmées avec un téléphone ou
une autre caméra. Dans Spyder, ouvrez `scripts/squat_front_video.py`, puis
exécutez le fichier avec **F5**.

1. Choisissez la vidéo dans la fenêtre de sélection.
2. Indiquez à quelle seconde commence la posture debout immobile. Appuyez sur
   **Entrée** si elle commence au début de la vidéo.
3. Les trois secondes suivantes sont utilisées comme baseline.
4. Le reste de la vidéo est analysé de face et les résultats sont enregistrés
   dans un nouveau sous-dossier `fppa_front_view_*` de `results/`.

Le mode vidéo frontal produit les données FPPA, les déviations gauche et
droite, les répétitions, les vitesses du bassin, le cycle moyen, les graphiques
et le rapport clinique descriptif. Une vidéo `frontal_annotated.mp4` montre le
squelette, les segments hanche-genou-cheville, les FPPA et les déviations
calculées image par image. Les coordonnées sont corrigées selon les
dimensions réelles de l'image afin de conserver les mêmes mesures en portrait
et en paysage.

```bash
python scripts/squat_front_video.py
```

## Protocole de prise de vue

Pour comparer plusieurs séances, conservez autant que possible :

- le même côté du corps présenté à la caméra ;
- la caméra perpendiculaire au plan sagittal, fixe et non inclinée ;
- le corps visible en entier, notamment l'épaule, la hanche, le genou, la cheville, le talon et l'avant-pied ;
- une distance, une hauteur de caméra, un éclairage et des vêtements similaires ;
- une posture debout immobile pendant la baseline ;
- une consigne de squat et une cadence comparables.

Pour une vidéo enregistrée sur téléphone, commencez l'enregistrement avant la baseline, gardez le téléphone fixe pendant tout le test et évitez de changer de zoom ou d'orientation. Le fichier vidéo conserve sa propre chronologie : les vitesses sont calculées à partir des timestamps de la vidéo et non de la vitesse de traitement de l'ordinateur.

Les mesures 2D sont surtout pertinentes pour suivre une même personne dans des conditions standardisées. Elles ne doivent pas être assimilées directement à une mesure instrumentale 3D.

## Résultats générés

Chaque essai crée un sous-dossier horodaté dans `results/`. Selon la qualité de la session, il peut contenir :

- les données sagittales brutes et traitées au format CSV ;
- les bornes des répétitions et les métriques par répétition ;
- le rapport sagittal au format texte ;
- le graphique temporel de la session ;
- le cycle moyen, son résumé CSV et son graphique ;
- les métadonnées et contrôles qualité de la session.
- pour une vidéo importée, la vidéo de contrôle annotée.

Le dossier `results/` et les vidéos placées dans `data/videos/` ne sont pas suivis par Git afin d'éviter de publier des données de test ou potentiellement personnelles.

## Tests

Depuis la racine du projet :

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Organisation du code

- `scripts/` : points d'entrée des modules frontal et sagittal ;
- `movement_analysis/` : calcul des angles, métriques et cycles moyens ;
- `movement_segmentation/` : détection des répétitions ;
- `signal_processing/` : filtrage et traitement des signaux ;
- `protocol/` et `quality/` : préparation et contrôles qualité ;
- `reporting/` : sauvegarde, rapports et graphiques ;
- `app/`, `session/`, `ui/` et `vision/` : acquisition et orchestration ;
- `app/side_video_pipeline.py` : adaptation d'une vidéo enregistrée au pipeline sagittal ;
- `tests/` : tests automatisés.

Cette séparation vise à garder les calculs biomécaniques indépendants de la webcam. L'application mobile pourra ainsi réutiliser les règles métier et remplacer progressivement la couche d'acquisition et l'interface du prototype.

## Limites actuelles et suite prévue

- validation sur davantage de personnes, morphologies, vêtements, éclairages et appareils ;
- formalisation d'un protocole standard et de critères d'acceptation ;
- amélioration des indicateurs de confiance et de qualité ;
- validation des seuils avant toute recommandation destinée aux professionnels ;
- définition d'un format de données stable pour le futur client mobile ;
- conception de l'expérience mobile et gestion responsable des données personnelles.

## Licence

Ce projet est distribué sous licence MIT. Consultez le fichier `LICENSE`.
