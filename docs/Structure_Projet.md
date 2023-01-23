# Structure des projets

Un nouveau projet doit minimalement contenir :

1. Un fichier [README.md](../README.md) principal qui décrit le projet et qui contient :
   1. Les auteurs du projet et leurs coordonnées (GitHub)
   2. Une description courte
   3. Un lien vers:
      1. Les instructions de compilation
      2. Les instructions d'installation
      3. Les exemples d'utilisation
      4. La documentation
   4. Les publications importantes
      1. Site Web
      2. Articles scientifiques
      3. Vidéos YouTube
   5. Remerciements
      1. Un logo du laboratoire / compagnie / organisation et lien Internet
2. Un fichier [LICENSE](../LICENSE) qui mentionne clairement la license des fichiers sources.
   1. Ex: GPLv3, BSD 3 clauses, etc.
3. Une description sommaire pour facilier la recherche (administratinon du projet GitHub)
4. Les mots clés du projet (administratinon du projet GitHub)
5. Les ["workflows"](../.github/workflows) pour l'intégration continue
6. Une organisation adéquate des répertoires (selon le type projet) et une explication de l'organisation.

## Projets C/C++

* Utilisation de [CMake](https://cmake.org/) comme "build system"
* Style de code uniforme. Ex: [Google](https://google.github.io/styleguide/cppguide.html)
* Utilisation des standards modernes C++11/14/20
* Tests avec [GoogleTest](https://github.com/google/googletest)
* Distribution par release / installateur (si possible)

## Projets Python

* Conformité au style [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Tests avec la bibliothèque intégrée [unittest](https://docs.python.org/3/library/unittest.html)
* Distribution avec [pypi.org](https://pypi.org/) (si possible)
