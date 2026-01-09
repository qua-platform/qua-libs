# Contributing

This repository contains libraries of files demonstrating the functionality of QUA and showing examples of its uses.
Did you write nice code which can be useful for others? Have an idea for something to add? We'd love to have your contribution!
There are several ways in which you can contribute, please see what suits you the most and don't be shy about asking for help.

## Opening an issue

The easiest way to contribute is to simply open an [issue](https://github.com/qua-platform/qua-libs/issues).
This can be used if you made a nice experiment you want to brag about, or if you found an issue in any of the existing examples.
In addition, this can also be used in case you're having any issue contributing your own code.

## Directly contributing to the repository

If you want to contribute code to the Python package, you would need to work with GitHub. 
If you've never worked with a git repository, then you should know that it is an online repository that helps with code management.
If you don't have git software on your computer, then you can use [GitHub Desktop](https://desktop.github.com/).
You can read more about git [here](https://docs.github.com/en/get-started/using-git/about-git).

GitHub has a [page](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) dedicated to explaining the steps for contributing to repositories. 
You can click on the desktop link on the top of the webpage to get to the GitHub Desktop instructions.
The steps can be summarized as follows:
1. Fork the repository
2. Create a branch (Optional, see below)
3. Write your changes and push them.
4. Open a Pull Request (PR) back to the main repository.
5. An admin will take a look at your code. If changes are needed, we will either ask you to do them or, with your permission, do them ourselves.
6. An admin will merge your contribution.

Feel free to ask for help at any step along the way. If needed, we will step in and help you finalize your code.

### Code standards

In order to maintain a high standard of quality, we follow several coding guidelines, and we have a few tips to help facilitate the process.
At any point, you can simply open a PR, state that you are having problems with the technical details, and we would step in and help.

Coding tips to avoid [merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github):
1. Build new features or fix existing ones in a branch. You can do as many commits as you want to a branch, saving your work as you go along. You can push this feature branch to save your work and back it up. 
   
    Note: You must work on your own fork, we recommend that you create a branch in your fork.
2. Make sure your [fork is in sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) with the main repository.

Code standard guidelines:
1. When you are ready to merge your code into the main repository, you need to make sure that all of your commits and commit messages are good and descriptive. You can always squash your commits into a few (or a single) commits.
2. We use [black](https://pypi.org/project/black/) to make sure that our code is standardized. Format your code before opening the PR by first install black:```pip install black``` and then type in the terminal ```black .``` (At the root of the repository)

### Development Setup (Recommended)

For the best development experience, we recommend using `uv` for dependency management and setting up pre-commit hooks.

#### Install Dependencies

Using uv (recommended):
```bash
uv sync --group dev --prerelease=allow
```

Or using Poetry (legacy):
```bash
poetry install
```

#### Set up Pre-commit Hooks

Pre-commit hooks automatically check your code for formatting and linting issues before each commit.

Install pre-commit hooks:
```bash
uv run pre-commit install
```

Run pre-commit manually on all files:
```bash
uv run pre-commit run --all-files
```

#### Running Tests

```bash
uv run pytest
```

#### Code Formatting

Check formatting:
```bash
uv run black --check .
```

Format code:
```bash
uv run black .
```

#### Linting

Run pylint to check code quality:
```bash
uv run pylint <your_file_or_directory>
```

# Contributor License Agreement

Submitting code to this project is conditioned upon all contributors to signing a contributor license agreement.
Agreeing to the contributor license agreement (CLA) means you declare that you are the author of the contribution and 
that you're freely contributing it under the terms of the CLA.

The [individual CLA](CLA/QUA_SDK_libraries.pdf)
document is available for review as a PDF.

**Note**:
> If your contribution is part of your employment or your contribution
> is the property of your employer, then you will likely need to sign a
> [corporate CLA](CLA/QUA_SDK_libraries_Corporate.pdf) too and
> email it to us at <gal@quantum-machines.co>.
