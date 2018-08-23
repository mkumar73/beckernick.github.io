---
layout: archive
permalink: /timeseries/
title: "Time Series posts by tags"
author_profile: true
header:
  image: "neural-nets-image.jpeg"
  caption: "photo credit: https://cdn-images-1.medium.com/"

---

{% include base_path %}
{% include group-by-array collection=site.portfolio field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}