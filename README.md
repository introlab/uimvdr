# demo_integration

[![Actions Status](https://github.com/introlab/demo_integration/actions/workflows/build_test_and_release.yml/badge.svg)](https://github.com/introlab/demo_integration/actions)

Projet de démonstration pour l'intégration, distribution et déploiement continus (CI/CD) avec Git et les outils disponibles sur GitHub. Nous proposons aussi une structure de projet pour uniformiser la présentation et la manière de faire pour tous nos projets.

L'objectif de demo_integration est donc de montrer, par un exemple concret, les bonnes pratiques à mettre en place pour les projets du laboratoire IntRoLab. Ce projet peut également servir de point de départ pour la création d'un nouveau projet.

Dans le répertoire racine, vous trouverez :

* [Une bibliothèque Python](library/pycount)
* [Une bibliothèque C++](library/cppcount)
* [Les bindings Python de la bibliothèque C++](library/cppcount/cppcount/python)
* [Un "benchmark" des trois bibliothèques](library/benchmarks)
* Les tests unitaires pour chaque bibliothèque
* [Un firmware Arduino](firmware)
* Les tests de compilation du firmware
* Les [scripts](.github/workflows) pour automatiser la compilation, les tests et les déploiements avec les [actions GitHub](https://docs.github.com/en/actions)

## Obtenir le code

```bash
# Obtention du code avec tous les composants (sous-modules)
git clone https://github.com/introlab/demo_integration.git --recurse-submodules
```

## Documentation

* [Introduction à l'intégration continue](docs/Introduction_Integration_Continue.md)
* [Outils utilisés](docs/Outils.md)
* [Structure des projets](docs/Structure_Projet.md)
* [Flux de travail](docs/Flux_Travail.md)

## Améliorations

Faites-nous part de vos commentaires / suggestions pour améliorer le projet dans les ["Issues"](https://github.com/introlab/demo_integration/issues).

## Auteurs

* Marc-Antoine Maheux (@mamaheux)
* Dominic Létourneau (@doumdi)

## Licence

* [BSD-3](LICENSE)

## Remerciements

![IntRoLab](docs/IntRoLab.png)

[IntRoLab - Laboratoire de robotique intelligente / interactive / intégrée / interdisciplinaire @ Université de Sherbrooke](https://introlab.3it.usherbrooke.ca)
