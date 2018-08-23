---
title: "About"
permalink: /about/

---

Manish is currently a research student at Universität Osnabrück in the field of 
Cognitive Science. His main area of research is Neural networks particularly 
Convolutional Neural Networks(CNN), Autoencoder(AE), RNN and LSTM. 
Currently, he is working on applications of CNN for image classification and 
object detection. He has over 6 years of industrial experience in building machine 
learning frameworks, data analytics solutions for Pharmaceuticals, Healthcare and 
Energy sectors. He has worked intensively on advanced machine learning and 
optimization problem. He was also part of building an Open Source Data Science 
and Big Data platform named KAVE. With practical industry experience and immense 
motivation for research in advance Neural Networks and Machine Learning, he is 
actively involved in Deep Learning project and wishes to contribute in this field 
of science.


{% include base_path %}
{% include group-by-array collection=site.portfolio field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}