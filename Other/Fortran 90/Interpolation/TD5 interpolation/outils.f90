module outils
  use nr, only: splint
  use donnees_communes
  implicit none
contains
	function func(x)
	  real(4), intent(in) :: x
	  real(4) :: func

	  ! Affichage pour voir la valeur de x
!	  print *, "Entrée de func: x = ", x
	  
	  ! Calcul de la puissance I * V
	  func = -x * splint(List_x, List_y, List_y2, x)

	  ! Affichage pour vérifier le résultat de splint
!	  print *, "Résultat de splint pour x = ", x, " est ", func
	  
	end function func
end module outils




