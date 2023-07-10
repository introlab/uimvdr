# Outils

Voici une liste des outils que nous utilisons pour ce projet. Il est recommandé d'en faire la lecture et d'en maîtriser les bases.

## Git

[Git](https://www.atlassian.com/fr/git/tutorials/what-is-version-control) est un outil de contrôle de source distribué qui permet aux développeurs de gérer les changements de façon organisée. Plus particulièrement, vous devriez vous familiariser avec les opérations suivantes :

* [Clone](https://www.atlassian.com/fr/git/tutorials/setting-up-a-repository/git-clone) permet de créer une copie locale d'un dépôt pour travailler.
* [Fetch](https://www.atlassian.com/fr/git/tutorials/syncing/git-fetch). Télécharge le contenu du dépôt distant sans modifier le contenu local.
* [Merge](https://www.atlassian.com/fr/git/tutorials/using-branches/git-merge) Intégration des changements d'une branche vers une autre.
* [Pull](https://www.atlassian.com/fr/git/tutorials/syncing/git-pull) est une combinaison de fetch d'une branche et merge dans la branche de travail.
* [Push](https://www.atlassian.com/fr/git/tutorials/syncing/git-push) publie les changements locaux vers un dépôt central (remote). Ceci rend disponible vos changements à tous les développeurs.
* [Branch](https://www.atlassian.com/fr/git/tutorials/using-branches) création d'une "ligne" de développement indépendante qui pourra être "mergé" en temps voulu.
* [Checkout](https://www.atlassian.com/fr/git/tutorials/using-branches/git-checkout) permet de basculer d'une branche à l'autre.
* [Commit](https://www.atlassian.com/fr/git/tutorials/saving-changes/git-commit) Enregistrement (versionnement) local d'un ou de plusieurs fichiers.

Git est à la base un outil utilisé sur la ligne de commande. Par contre, il existe des outils (GUI) pour faciliter son utilisation. En voici quelques-un:

* [GitHub Desktop](https://desktop.github.com/)
* [Visual Studio Code](https://code.visualstudio.com/download)
* [GitKraken](https://www.gitkraken.com/download)
* [TortoiseGit](https://tortoisegit.org/)

Il y en a plusieurs autres. À vous de choisir l'outil qui vous convient!

### Sous-modules Git (Git Submodules)

Les [sous-modules git](https://www.atlassian.com/fr/git/tutorials/git-submodule) sont très utiles pour référencer d'autres dépôts git qui sont souvent des dépendences (libraries, outils, documentation, etc.) à un projet.
De cette façon, il est possible de bien contrôler les versions des dépendances à partir du dépôt "maître".
Cette outil est souvent utilisé pour les projets C++ avec des bibliothèques à compiler.
Vous pouvez voir un exemple d'utilisation de sous-modules pour ce projet [ici](../library/cppcount/3rdParty).
Par contre, pour certains projets, il est préférable de favoriser les gestionnaires de dépendences comme NPM ou pip au lieu de faire des submodules puisqu'ils font normalement référence à des versions "distribuées" et supportées.

## CMake

[CMake](https://cmake.org/cmake/help/v3.20/) est un outil pour la gestion de compilation des sources.
Il s'agit en fait de décrire la recette (ou procédure) à suivre pour la compilation du projet.
Ceci est effectué à partir d'un interpréteur (cmake) qui génère des fichiers (Makefiles, projet XCode, Visual Studio, etc.) à partir de scripts (CMakeFiles.txt).
Une bonne maîtrise de cet outil devient essentiel dans les projets qui sont plus complexes et qui nécessitent la compilation de plusieurs composants.

## GitHub Issues

GitHub nous permet de créer des ["Issues"](https://github.com/introlab/demo_integration/issues) à partir de leur portail et de [référencer](https://docs.github.com/en/github/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls) chacune d'elle dans les commentaires de commits.
Ceci sera utile pour faciliter le suivi des projets.

## GitHub Actions

[GitHub Actions](https://docs.github.com/en/actions) Permet d'automatiser les "flux" (workflows) de travail pour permettre [l'intégration continue](Introduction_Integration_Continue.md).
Il est possible de personnaliser le comportement des actions selon plusieurs "événements" tels les push/pull-requests dans certaines branches de votre projet. 
Ceci est utile par exemple lorsque vous faites un pull-request dans la branche "main" pour vérifier le code avec des tests avant de faire le merge.
Vous pouvez consulter les "workflows" de ce projet [ici](../.github/workflows).
