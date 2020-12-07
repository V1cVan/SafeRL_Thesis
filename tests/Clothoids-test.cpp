#include "ClothoidList.hh"

int main(){
    G2lib::ClothoidList cl = G2lib::ClothoidList();
    cl.push_back(0,0,0,-0.5,1,50);
    cl.push_back(1,-0.4,50);
    cl.push_back(0.6,-0.6,50);
    double th,k,x,y;
    cl.evaluate(105,th,k,x,y);
    return 0;
}