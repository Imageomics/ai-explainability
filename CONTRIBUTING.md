Thank you for contributing!

This document outlines guidelines for collaboratively contributing to this repository (repo). It follows a branch and pull request (PR) based workflow, which provides a controlled way to bring internal contributions together for those with write access to the repository (those without write access will need to fork the repository first before making contributions).

Importantly, this workflow suggests that contributions are created through PRs rather than directly committing to or merging into the `main` branch.

**To make a contribution, please follow these steps:**
1. **Clone the repo to your computer.**
```
git clone https://github.com/Imageomics/ai-explainability.git
cd ai-explainability
```
2. **Create a new branch.** For example, if you want to add a feature to your code that simulates human vision, you could name the branch `feature/simulate-vision`. (_pro-tip_, make a new branch for each PR scoped by the task, feature, or bug fix).
```
git branch feature/simulate-vision
git checkout feature/simulate-vision
```
or to create and switch to the new branch with a single command:
```
git checkout -b feature/simulate-vision
```
3. **Make your desired changes.** For example, imagine you created three new files, each simulating a component of the human visual system: `retina.py`, `occipital.py`, and `visual_cortex.py`.
4. **Commit changes to the new branch.** (_pro-tip_, commit frequently with each commit based on a logical self-contained change using descriptive commit messages. _pro-tip_, use imperative phrases beginning with words such as "add", "update", "fix", "refactor", "remove", "improve", ...).
```
git commit -m "Implement the retina, occipital, and visual cortex components of the human visual system."
```
5. **Update your local `main` branch.** Ensure your local `main` branch is up-to-date with the remote to incorporate any changes other collaborators may have made. (_pro-tip_, if you're unsure what branch you should have checked out, remember that the branch being merged to or committed to should be the branch that is active. Check with `git branch` and look for `*` next to what's active.)
```
git checkout main
git pull origin main
```
6. **Merge changes made to `main` to your new branch.** If updates were pulled into your local `main` branch, merge them into your new branch.
```
git checkout feature/simulate-vision
git merge main
```
7. **Push your new branch to the remote.** This should contain any updates made by others as well as your new changes.
8. **Make, commit, and push with this branch as needed.** Repeat steps 3-7 until results are in a state suitable to merge with the project's `main` branch.
9. **Open a Pull Request.** On the GitHub repo page, click the "Pull requests tab, click the "New pull requests" button, select the new branch you pushed as the head branch and keep the base branch as `main` (where you want to merge your changes into). Click "Create pull request. You can also use draft PRs to solicit feedback before opening a real PR. You can also consider using the [GitHub CLI]([url](https://cli.github.com/)) for this step. (_pro-tip_, keep PRs small and manageable for review; the scope should be focused on the task, feature, or bug fix associated with the branch)
10. **Verify the repositories and branches in the PR.** Base Repository: The original repo you have write access to. Head Repository: The same repo. Base Branch: `main` (or the branch you want to merge your changes into) Compare Branch: Your new branch with changes.
11. **Title and describe the PR.** Optionally assign reviewers and link the PR to a project.
12. **Submit the PR.** "Click "Create pull request once more to submit.

After a branch is merged and a PR is closed, delete the branch from the remote and your local repository to keep things tidy. 
