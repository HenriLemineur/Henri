subroutine solve(ndim, n, a, b, ipvt)

  use nrtype

  implicit none
  integer, intent(IN) :: ndim, n
  integer, intent(IN), dimension(:) :: ipvt
  real (DP), intent(IN), dimension(:,:) :: a
  real (DP), intent(IN OUT), dimension(:) :: b

!c   solution of linear system, a*x = b .
!c   do not use if decomp has detected singularity.
!c
!c   input..
!c
!c     ndim = declared row dimension of array containing a .
!c
!c     n = order of matrix.
!c
!c     a = triangularized matrix obtained from decomp .
!c
!c     b = right hand side vector.
!c
!c     ipvt = pivot vector obtained from decomp .
!c
!c   output..
!c
!c     b = solution vector, x .
!c
  integer ::  kb, km1, nm1, kp1, i, k, m
  real (DP) :: t
!c
!c     forward elimination
!c
  if (ndim .eq. n) go to 1
1  if (n .eq. 1) go to 50
  nm1 = n-1
  do k = 1, nm1
     kp1 = k+1
     m = ipvt(k)
     t = b(m)
     b(m) = b(k)
     b(k) = t
     do i = kp1, n
        b(i) = b(i) + a(i,k)*t
     end do
  end do
!c
!c     back substitution
!c
  do kb = 1,nm1
     km1 = n-kb
     k = km1+1
     b(k) = b(k)/a(k,k)
     t = -b(k)
     do i = 1, km1
        b(i) = b(i) + a(i,k)*t
     end do
  end do
50 b(1) = b(1)/a(1,1)
  return
end subroutine solve
