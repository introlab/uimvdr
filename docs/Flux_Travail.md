# Flux de travail

Voici le flux de travail qui est proposé:

1. Création d'un [Issue](https://github.com/introlab/demo_integration/issues) pour référencer un problème ou une amélioration.
2. Création d'une [branche](https://www.atlassian.com/fr/git/tutorials/using-branches) à partir de la branche "main" pour travailler sur l'amélioration.
   1. S'assurer que votre branche "main" est à jour par rapport au dépôt principal sur GitHub avant de créer votre branche en faisant un [pull](https://www.atlassian.com/fr/git/tutorials/syncing/git-pull).
3. Faire les changements dans la nouvelle branche et s'assurer que les tests fonctionnent.
   1. Les commentaires des "commits" peuvent faire référence à ce Issue : Ex "Ref #234 Ajout de fonctionnalité x."
4. Faire un [push](https://www.atlassian.com/fr/git/tutorials/syncing/git-push) sur le dépôt GitHub dans une branche du même nom.
   1. Vous pouvez faire plusieurs "push" dans cette branche si le travail prend plus d'une journée. Ceci permet aux autres développeurs de voir vos progrès et de vous assister en cas de besoin.
5. Quand vous êtes satisfait(e)s des modifications apportés et quand vous avez terminé l'amélioration, faites un [pull-request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) dans la branche principale "main" à partir des outils de GitHub.
   1. Vous pouvez nommer un reviewer
   2. **Vos "workflows" devraient exécuter automatiquement les tests à ce moment pour vérifier le code.**
   3. Il n'est pas nécessaire de refaire un pull-request si vous étiez déjà en train de modifier une branche dans laquelle le pull-request a déjà été demandé.
6. Les "reviewers" peuvent commenter les changements et demander des modifications. Si c'est le cas, retournez à l'étape 3 pour compléter les changements. Si tout est accepté, passez à la prochaine étape.
7. La personne responsable du projet peut exécuter le "merge" dans la branche "main".
   1. Faire un [squash-and-merge](https://docs.github.com/en/github/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges) pour éviter les commits intermédiaire qui ne fonctionnent pas nécessairement.
   2. **Vos workflows devraient effectuer les tests, et le déploiement de la nouvelle version.**
   3. Si tout s'est bien passé, la branche qui vient d'être "mergé" peut être effacée.
