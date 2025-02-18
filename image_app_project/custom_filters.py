from django import template

register = template.Library()

@register.filter(name='lowercase')
def lowercase(value):
    if isinstance(value, str):
        return value.lower()
    return value
