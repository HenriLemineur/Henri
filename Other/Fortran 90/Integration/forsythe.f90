module forsythe
  interface
     subroutine quanc8(fun,a,b,abserr,relerr,result,errest,nofun,flag)
       use nrtype
       implicit none
       real(dp), intent(in) :: a, b, abserr, relerr
       real(dp), intent(out) :: result, errest, flag
       integer, intent(out) :: nofun
       interface
          function fun(x)
            use nrtype
            implicit none
            real(dp), intent(in) :: x
            real(dp) :: fun
          end function fun
       end interface
     end subroutine quanc8
  end interface
  interface
     SUBROUTINE decomp(ndim,n,a,cond,ipvt,work)
       USE nrtype
       IMPLICIT NONE
       INTEGER, INTENT(IN) :: ndim,n
       REAL (DP), INTENT(IN OUT), DIMENSION(:,:) :: a  
       REAL (DP), INTENT(OUT) :: cond
       REAL (DP), INTENT(IN OUT), DIMENSION(:) :: work
       INTEGER, INTENT(IN OUT), DIMENSION(:) :: ipvt
     end SUBROUTINE decomp
  end interface
  interface 
     subroutine solve(ndim, n, a, b, ipvt)
       use nrtype
       implicit none
       integer, intent(IN) :: ndim, n
       integer, intent(IN), dimension(:) :: ipvt
       real (DP), intent(IN), dimension(:,:) :: a
       real (DP), intent(OUT), dimension(:) :: b
     end subroutine solve
  end interface

interface
   subroutine rkf45(f,neqn,y,t,tout,relerr,abserr,iflag,work,iwork)
     use nrtype
     implicit none
     integer, intent(in) :: neqn
     integer, intent(in out) :: iflag
     integer, intent(in out),dimension(:) :: iwork
     real (dp), intent(in out), dimension(:) :: y
     real (dp), intent(in out) :: t
     real (dp), intent(in) :: tout
     real (dp), intent(in out) :: relerr,abserr
     real (dp), intent(in out),dimension(:) :: work
     
     interface
        subroutine f(t,y,yp)
          use nrtype
          implicit none
          real(dp), intent(in) :: t
          real(dp), intent(in),dimension(:) :: y
          real(dp), intent(out),dimension(:) :: yp
        end subroutine f
     end interface
   end subroutine rkf45
end interface

interface
   subroutine rkfs(f,neqn,y,t,tout,relerr,abserr,iflag,yp,h,f1,f2,f3,&
        f4,f5,savre,savae,nfe,kop,init,jflag,kflag)
     use nrtype
     implicit none
     integer, intent(in) :: neqn
     integer, intent(in out) :: iflag,kop,init,jflag,kflag
     integer, intent(out) :: nfe
     real (dp), intent(in out), dimension(:) :: y,yp,f1,f2,f3,f4,f5
     real (dp), intent(in out) ::  t,h,savre,savae,relerr,abserr
     real (dp), intent(in) :: tout
     interface
        subroutine f(t,y,yp)
          use nrtype
          implicit none
          real(dp), intent(in) :: t
          real(dp), intent(in),dimension(:) :: y
          real(dp), intent(out),dimension(:) :: yp
        end subroutine f
     end interface
   end subroutine rkfs
end interface

interface
   subroutine fehl(f,neqn,y,t,h,yp,f1,f2,f3,f4,f5,s)
     use nrtype
     implicit none
     integer, intent(in) :: neqn
     real (dp), intent (in out), dimension(:) ::  y,yp,f1,f2,f3,f4,f5,s
     real (dp), intent(in) :: h,t
     interface
        subroutine f(t,y,yp)
          use nrtype
          implicit none
          real(dp), intent(in) :: t
          real(dp), intent(in),dimension(:) :: y
          real(dp), intent(out),dimension(:) :: yp
        end subroutine f
     end interface
   end subroutine fehl
end interface

end module forsythe
