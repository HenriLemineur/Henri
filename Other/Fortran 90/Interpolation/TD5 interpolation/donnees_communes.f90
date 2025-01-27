module donnees_communes
  implicit none
  integer, parameter :: max_points = 100  ! Nombre maximal de points
  integer :: n  ! Nombre de points lus
  real(4), allocatable :: List_x(:)  ! Vecteur des courants (allouable)
  real(4), allocatable :: List_y(:)  ! Vecteur des tensions (allouable)
  real(4), allocatable :: List_y2(:)  ! Vecteur des dérivées secondes pour l'interpolation spline
  real(4) :: yp1 = 1.0E30_4, ypn = 1.0E30_4  ! Valeurs des dérivées premières (pour une spline naturelle)
end module donnees_communes
