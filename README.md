# Welcome to branch #1 - CLDK notebooks in Colab

This was about getting the CLDK notebooks to run in Colab, I (Trevor) eventually
gave it all up as a bad job. You can see the work in `examples/code_summarization_colab.ipynb`.

It's super hacky for reasons beyond my control- which I will enumerate now. 

- [ ] Colab needed to have python bumpoed to 3.11 (this is a CLDK requirement). I have code that will do this, but it requires restarting the kernel, and ultimately is hacky.
- [ ] Numpy issues- ollama community and CLDk are stuck in numpy 1.x but colab ships with 2.x. Again, can be hacked around, but hacky.

So the reccomendation is to keep an eye on Colab, CLDK, and ollama-community and wait for them to make required updates, then revisit this endevor.
