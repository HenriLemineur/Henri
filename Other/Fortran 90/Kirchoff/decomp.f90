SUBROUTINE decomp(ndim,n,a,cond,ipvt,work)

  USE nrtype

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: ndim,n
  REAL (DP), INTENT(IN OUT), DIMENSION(:,:) :: a  
  REAL (DP), INTENT(OUT) :: cond
  REAL (DP), INTENT(IN OUT), DIMENSION(:) :: work
  INTEGER, INTENT(IN OUT), DIMENSION(:) :: ipvt

  !c     decomposes a double precision matrix by gaussian elimination
  !c     and estimates the condition of the matrix.
  !c
  !c     use solve to compute solutions to linear systems.
  !c
  !c     input..
  !c
  !c        ndim = declared row dimension of the array containing  a.
  !c
  !c        n = order of the matrix.
  !c
  !c        a = matrix to be triangularized.
  !c
  !c     output..
  !c
  !c        a  contains an upper triangular matrix  u  and a permuted
  !c          version of a lower triangular matrix  i-l  so that
  !c          (permutation matrix)*a = l*u .
  !c
  !c        cond = an estimate of the condition of  a .
  !c           for the linear system  a*x = b, changes in  a  and  b
  !c           may cause changes  cond  times as large in  x .
  !c           if  cond+1.0 .eq. cond , a is singular to working
  !c           precision.  cond  is set to  1.0d+32  if exact
  !c           singularity is detected.
  !c
  !c        ipvt = the pivot vector.
  !c           ipvt(k) = the index of the k-th pivot row
  !c           ipvt(n) = (-1)**(number of interchanges)
  !c
  !c     work space..  the vector  work  must be declared and included
  !c                   in the call.  its input contents are ignored.
  !c                   its output contents are usually unimportant.
  !c
  !c     the determinant of a can be obtained on output by
  !c        det(a) = ipvt(n) * a(1,1) * a(2,2) * ... * a(n,n).
  !c

  REAL (DP) :: ek, t, anorm, ynorm, znorm
  INTEGER :: nm1, i, j, k, kp1, kb, km1, m
  
  INTERFACE
     SUBROUTINE solve(ndim, n, a, b, ipvt)
       USE nrtype
       IMPLICIT NONE       
       INTEGER, INTENT(IN):: ndim,n
       INTEGER, INTENT(IN), dimension(:) :: ipvt
       REAL(DP), INTENT(IN), dimension(:,:) :: a
       REAL(DP), INTENT(OUT), dimension (:) :: b
     END SUBROUTINE solve
  END INTERFACE

  ipvt(n) = 1
  IF (n .EQ. 1) go to 80
  nm1 = n - 1
  !c
  !c     compute 1-norm of a
  !c
  anorm = 0.0d0
  DO j = 1, n
     t = 0.0d0
     DO i = 1, n
        t = t + abs(a(i,j))
     END DO
     IF (t .GT. anorm) anorm = t
  END DO
  !c
  !c     gaussian elimination with partial pivoting
  !c
  DO k = 1,nm1
     kp1= k+1
     !c
     !c        find pivot
     !c
     m = k
     DO i = kp1,n
        IF (abs(a(i,k)) .GT. abs(a(m,k))) m = i
     END DO
     ipvt(k) = m
     IF (m .NE. k) ipvt(n) = -ipvt(n)
     t = a(m,k)
     a(m,k) = a(k,k)
     a(k,k) = t
     !c
     !c        skip step if pivot is zero
     !c
     IF (t .EQ. 0.0d0) go to 35
     !c
     !c        compute multipliers
     !c
     DO i = kp1,n
        a(i,k) = -a(i,k)/t
     END DO
     !c
     !c        interchange and eliminate by columns
     !c
     DO j = kp1,n
        t = a(m,j)
        a(m,j) = a(k,j)
        a(k,j) = t
        IF (t .EQ. 0.0d0) go to 30
        DO i = kp1,n
           a(i,j) = a(i,j) + a(i,k)*t
        END DO
30   END DO
35  END DO
  !c
  !c     cond = (1-norm of a)*(an estimate of 1-norm of a-inverse)
  !c     estimate obtained by one step of inverse iteration for the
  !c     small singular vector.  this involves solving two systems
  !c     of equations, (a-transpose)*y = e  and  a*z = y  where  e
  !c     is a vector of +1 or -1 chosen to cause growth in y.
  !c     estimate = (1-norm of z)/(1-norm of y)
  !c
  !c     solve (a-transpose)*y = e
  !c
  DO k = 1, n
     t = 0.0d0
     IF (k .EQ. 1) go to 45
     km1 = k-1
     DO i = 1, km1
        t = t + a(i,k)*work(i)
   END DO
45   ek = 1.0d0
     IF (t .LT. 0.0d0) ek = -1.0d0
     IF (a(k,k) .EQ. 0.0d0) go to 90
     work(k) = -(ek + t)/a(k,k)
  END DO
  DO kb = 1, nm1
     k = n - kb
     t = 0.0d0
     kp1 = k+1
     DO i = kp1, n
        t = t + a(i,k)*work(i)
     END DO
     work(k) = t + work(k)
     m = ipvt(k)
     IF (m .EQ. k) go to 60
     t = work(m)
     work(m) = work(k)
     work(k) = t
60 END DO

  ynorm = 0.0d0
  DO i = 1, n
     ynorm = ynorm + abs(work(i))
 END DO
  !c
  !c     solve a*z = y
  !c
  CALL solve(ndim, n, a, work, ipvt)
  !c
  znorm = 0.0d0
  DO i = 1, n
     znorm = znorm + abs(work(i))
 END DO
  !c
  !c     estimate condition
  !c
  cond = anorm*znorm/ynorm
  IF (cond .LT. 1.0d0) cond = 1.0d0
  RETURN
  !c
  !c     1-by-1
  !c
80 cond = 1.0d0
  IF (a(1,1) .NE. 0.0d0) RETURN
  !c
  !c     exact singularity
  !c
90 cond = 1.0d+32
  RETURN
END SUBROUTINE decomp