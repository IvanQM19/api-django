from django.urls import path
from . import views
from . import viewsTwo
from . import viewsThree
from . import viewsTwo
from . import viewsForr
from . import viewsFive

urlpatterns = [
    path('', views.index, name='index'),  # Ruta para la vista index en views.py
    path('section14/', views.section14, name='section14'),  # Ruta para la vista section14 en viewsTwo.py
    path('section15/', viewsTwo.section15, name='section15'),
    path('section16/', viewsThree.section16, name='section16'),
    path('section17/', viewsForr.index, name='section17'),   # Ruta para la vista section16 en viewsThree.py
    path('section18/', viewsFive.section18, name='section18'),    # Ruta para la vista section16 en viewsThree.py
    # Otras rutas...
]
