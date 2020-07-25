package com.nju.parser;

import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class APISequenceParser {
    public String parserMethodBody(BlockStmt bStmt) {
        String apiSeq = "";
        if (!bStmt.getStatements().isEmpty()) {
            List<String> apiList = new ArrayList<>();
            VoidVisitor<List<String>> methodCallVisitor = new MethodCallVisitor();
            methodCallVisitor.visit(bStmt, apiList);
            apiSeq = String.join("->", apiList);
        }
        return apiSeq;
    }

    private static class MethodCallVisitor extends VoidVisitorAdapter<List<String>> {
        @Override
        public void visit(MethodCallExpr methodCall, List<String> collector) {
            List<Expression> args = methodCall.getArguments();
            if (args != null)
                handleExpressions(args, collector);
            String apiStr = (methodCall.getScope().isPresent() ? methodCall.getScope().get() : "this") +
                    "." + methodCall.getName();
            collector.add(apiStr);
        }

        private void handleExpressions(List<Expression> expressions, List<String> collector)
        {
            for (Expression expr : expressions)
            {
                if (expr instanceof MethodCallExpr)
                    visit((MethodCallExpr) expr, collector);
                else if (expr instanceof BinaryExpr)
                {
                    BinaryExpr binExpr = (BinaryExpr)expr;
                    handleExpressions(Arrays.asList(binExpr.getLeft(), binExpr.getRight()), collector);
                }
            }
        }
    }
}
