---
layout: archive
permalink: /neuralnets/
title: "Neural Networks posts by tags"
author_profile: true
header:
  image: "nn-image.jpg"
  caption: "photo credit: https://cdn-images-1.medium.com/"

---

{% include base_path %}
{% include group-by-array collection=site.portfolio_nn field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}