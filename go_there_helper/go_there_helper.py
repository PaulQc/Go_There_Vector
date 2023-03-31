# Variables et méthodes en support à l'application "GoThereVector"
# Paul Grenier
# 2023


class Messages(object):
    def __init__(self):
        """Définie les différents messages à fournir à l'utilisateur aux différentes étapes, soit:
         'initial', 'go', 'no_go' et 'fin'
         """
        self.message = {
            'initial': "Ici bas, l'espace de jeu de Vector <br> <b><em><u>Cliquez</u></em></b>" +
                       " dans l'image pour indiquer à Vector l'endroit où vous désirez qu'il se rendre",
            'go': "<b><em>Bien reçu !</em></b> Voyons si Vector est disponible pour votre requête <br>" +
                  " (s'il s'arrête, c'est bon signe : C'est qu'il écoute les directives !)",
            'no_go': "<b><em>Malheureusement</em></b>, Vector est présentement occupé à répondre <br>" +
                     " à une autre demande. Actualiser (reloader) la page web pour réessayer",
            'no_connection': "<b><em>Malheureusement</em></b>, Vector ne répond pas aux commandes <br>" +
                     " (il est têtu parfois !). Actualiser (reloader) la page web pour réessayer",
            'fin': "<em>Ça y est !</em> J'espère qu'il s'est bien rendu là où vous le désiriez.<br>" +
                   " Vous êtes de retour à la page initiale. Actualiser (reloader) la page web pour réessayer"
        }


class VectorStatus(object):
    def __init__(self):
        """Variable pour garder l'état de Vector, soit:
        'available' (= valeur initiale) et 'not_available' (si Vector est occupé ailleurs)
        """
        self.status = 'available'
        self.connection = ''  # 3 possibilité: active, close, unavailable

