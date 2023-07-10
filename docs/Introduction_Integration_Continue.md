# INTRODUCTION

Dans plusieurs projets modernes de développement logiciel, il est souvent question de CI/CD. Mais, qu'est-ce que c'est ? Les termes CI/CD signifient Continuous Integration / Continuous Delivery & Deployment. Ces termes sont traduits en français par l'intégration continue, distribution continue et déploiement continu. Il est alors question ici d'automatiser les processus de développement, la compilation, les tests, les mises-à-jour du code de référence jusqu'à la mise en production du logiciel. Des cycles rapides, itératifs et courts de développement sont ainsi souhaités.

![REDHAT:CI/CD Image](https://www.redhat.com/cms/managed-files/styles/wysiwyg_full_width/s3/ci-cd-flow-desktop_edited_0.png?itok=TzgJwj6p)

## L'intégration continue

Avec l'aide de [Git](https://www.atlassian.com/fr/git/tutorials/what-is-version-control) et des outils de gestion de projet fournis par GitHub ou externes ([Jira](https://www.atlassian.com/fr/software/jira)), il est possible de développer des logiciels selon la méthode [Agile](https://www.atlassian.com/fr/agile). La fusion des changements, souvent effectués par des ["pull-request"](https://www.atlassian.com/fr/git/tutorials/making-a-pull-request) à partir de [branches](https://www.atlassian.com/fr/git/tutorials/using-branches) dédiées à une fonctionnalité spécifique, permettra de mettre à jour le code source de référence en continue.

## La distribution continue

Les modifications apportées par les dévelopeurs sont automatiquement testées avant de faire un [merge](https://www.atlassian.com/fr/git/tutorials/using-branches/git-merge) sur la branche principale (souvent appelée master ou main). La branche "main" contient alors la nouvelle version que tous les développeurs doivent utiliser comme référence principale.

## Le déploiement continu

Dans un contexte de production, il est nécessaire de déployer les lociciels ou bibliothèques sous forme d'installateur ou de "packages" utilisables (pypi, package NPM, zip). Il est possible d'automatiser cette étape à chaque fois que le "merge" se produit dans la branche principale.

## Flux de travail

La section [Flux de travail](Flux_Travail.md) présente comment ceci peut se mettre en oeuvre avec les [outils](Outils.md) proposés.

## RÉFÉRENCES

* [RedHat - Quelle est la différence entre CI et CD ?](https://www.redhat.com/fr/topics/devops/what-is-ci-cd)
