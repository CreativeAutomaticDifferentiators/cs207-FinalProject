# Milestone 2: Group 6

Team Members: Kyra Ballard, Paul-Emile Landrin, Yaoyang Lin, Dan Park 

## Introduction

The project is about automatic differentiation. In most modern numerical algorithms, people need to access the values of derivatives. Automatic differentiation is a method that can be generalized to most function and it can compute the derivatives fast and accurately at machine precision.

## Background

The idea of automatic differentiation is to decompose a function into elementary function. To do so, we use a chain of function composition. The forward mode enables to compute values of the function. We can compute the derivatives using the reverse mode or using dual numbers. Dual numbers are basically numbers that are real, the dual number squared is equal to zero. We can single out the derivative of a function by composing the Taylor series of elementary functions. 

## How to Use *OttoDiff*

The user installs the package, ideally with pip:
`pip install [--upgrade] OttoDiff`

Then the user need to check that he/she has the required packages with appropriate version (like numpy==1.16.3) in the requirements.txt.

When he/she wants to use it, the user needs to import the package:
```python
import OttoDiff
```

Then the user defines a function that needs to be differentiated and creates an function object from our package that takes into argument these function. 

## Software Organization

Directory structure:
```
OttoDiff/
	__init__.py
	README.md
	requirements.txt
	setup.py
	Test/
	Utility/
		forward.py
		reverse.py
		duals.py
		advanced_features.py
	Docs/
		milestone1.md
```

The names of the modules speak for their functionality. We use different modules to make it easier to work at the same time on different codes. We don’t have choose our idea for advanced feature yet. We will try to use the two test suite TravisCI and CodeCov then select only the best to use. We will distribute our package on Pypi, it is also a setup tool that will package our software. Hence users can use the pip install command.

## Implementation

We will use the basic data structures on python. We will implement a dual number class, and a function class. The function class will have attributes to evaluate the values of the function and of its derivatives. The dual class will have two attributes for the real and the imaginary part.

We will rely on numpy for some evaluations, notably for the elementary functions. To handle their derivatives, we will use two methods. The first one is with the dual numbers and we will use the well-known taylor series of this elementary functions. The second method is to use their symbolic derivatives which are also well known.
