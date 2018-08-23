---
layout: archive
permalink: /python/
title: "Python posts by tags"
author_profile: true
header:
  image: "python-pic.jpg"
  caption: "photo credit: manish kumar"

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