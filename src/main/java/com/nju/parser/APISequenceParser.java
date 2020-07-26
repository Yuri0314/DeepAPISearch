package com.nju.parser;

import com.github.javaparser.ast.ArrayCreationLevel;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class APISequenceParser {
    private List<String> apiList = new ArrayList<>();

    /**
     * 类的功能方法入口函数
     * @param bStmt
     * @return
     */
    public String parserMethodBody(BlockStmt bStmt) {
        if (!bStmt.getStatements().isEmpty())
            handleBlockStmt(bStmt);
        return apiList.isEmpty() ? "" : String.join("->", apiList);
    }

    private void handleStmt(Statement stmt) {
        if (stmt.isBlockStmt())
            handleBlockStmt(stmt.toBlockStmt().get());
        else if (stmt.isBreakStmt())
            handleBreakStmt(stmt.toBreakStmt().get());
        else if (stmt.isContinueStmt())
            handleContinueStmt(stmt.toContinueStmt().get());
        else if (stmt.isDoStmt())
            handleDoStmt(stmt.toDoStmt().get());
        else if (stmt.isExpressionStmt())
            handleExpressionStmt(stmt.toExpressionStmt().get());
        else if (stmt.isForEachStmt())
            handleForEachStmt(stmt.toForEachStmt().get());
        else if (stmt.isForStmt())
            handleForStmt(stmt.toForStmt().get());
        else if (stmt.isIfStmt())
            handleIfStmt(stmt.toIfStmt().get());
        else if (stmt.isLabeledStmt())
            handleLabeledStmt(stmt.toLabeledStmt().get());
        else if (stmt.isReturnStmt())
            handleReturnStmt(stmt.toReturnStmt().get());
        else if (stmt.isSwitchStmt())
            handleSwitchStmt(stmt.toSwitchStmt().get());
        else if (stmt.isSynchronizedStmt())
            handleSynchronizedStmt(stmt.toSynchronizedStmt().get());
        else if (stmt.isThrowStmt())
            handleThrowStmt(stmt.toThrowStmt().get());
        else if (stmt.isTryStmt())
            handleTryStmt(stmt.toTryStmt().get());
        else if (stmt.isWhileStmt())
            handleWhileStmt(stmt.toWhileStmt().get());
        else if (stmt.isYieldStmt())
            handleYieldStmt(stmt.toYieldStmt().get());
        else
            return;
    }

    private void handleBlockStmt(BlockStmt bStmt) {
        List<Statement> statList = bStmt.getStatements();
        for (Statement stmt : statList)
            handleStmt(stmt);
    }

    private void handleBreakStmt(BreakStmt breakStmt) {
        apiList.add("break");
    }

    private void handleContinueStmt(ContinueStmt continueStmt) {
        apiList.add("continue");
    }

    private void handleDoStmt(DoStmt doStmt) {
        apiList.add("do");
        handleStmt(doStmt.getBody());
        apiList.add("while");
        handleExpression(doStmt.getCondition());
    }

    private void handleExpressionStmt(ExpressionStmt expressionStmt) {
        handleExpression(expressionStmt.getExpression());
    }

    private void handleForEachStmt(ForEachStmt foreachStmt) {
        apiList.add("foreach");
        handleExpression(foreachStmt.getIterable());
        handleStmt(foreachStmt.getBody());
    }

    private void handleForStmt(ForStmt forStmt) {
        apiList.add("for");
        // 判断初始器
        for (Expression expression : forStmt.getInitialization())
            handleExpression(expression);
        // 判断条件语句
        Optional<Expression> compare = forStmt.getCompare();
        if (compare.isPresent())
            handleExpression(compare.get());
        // 判断更新语句
        for (Expression expression : forStmt.getUpdate())
            handleExpression(expression);
    }

    private void handleIfStmt(IfStmt ifStmt) {
        apiList.add("if");
        handleExpression(ifStmt.getCondition());
        handleStmt(ifStmt.getThenStmt());
        // 判断有无else语句
        Optional<Statement> elseStmt = ifStmt.getElseStmt();
        if (elseStmt.isPresent()) {
            apiList.add("else");
            handleStmt(elseStmt.get());
        }
    }

    private void handleLabeledStmt(LabeledStmt labeledStmt) {
        handleStmt(labeledStmt.getStatement());
    }

    private void handleReturnStmt(ReturnStmt returnStmt) {
        apiList.add("return");
        Optional<Expression> expression = returnStmt.getExpression();
        if (expression.isPresent())
            handleExpression(expression.get());
    }

    private void handleSwitchStmt(SwitchStmt switchStmt) {
        apiList.add("switch");
        handleExpression(switchStmt.getSelector());
        for (SwitchEntry entry : switchStmt.getEntries()) {
            for (Expression labelExpression : entry.getLabels())
                handleExpression(labelExpression);
            for (Statement stmt : entry.getStatements())
                handleStmt(stmt);
        }
    }

    private void handleSynchronizedStmt(SynchronizedStmt synchronizedStmt) {
        apiList.add("synchronized");
        handleExpression(synchronizedStmt.getExpression());
        handleBlockStmt(synchronizedStmt.getBody());
    }

    private void handleThrowStmt(ThrowStmt throwStmt) {
        apiList.add("throw");
        handleExpression(throwStmt.getExpression());
    }

    private void handleTryStmt(TryStmt tryStmt) {
        // 处理try语句块
        apiList.add("try");
        for (Expression expression : tryStmt.getResources())
            handleExpression(expression);
        handleBlockStmt(tryStmt.getTryBlock());
        // 处理catch语句
        apiList.add("catch");
        for (CatchClause clause : tryStmt.getCatchClauses())
            handleBlockStmt(clause.getBody());
        // 处理finally语句
        apiList.add("finally");
        Optional<BlockStmt> finallyStmt = tryStmt.getFinallyBlock();
        if (finallyStmt.isPresent())
            handleBlockStmt(finallyStmt.get());
    }

    private void handleWhileStmt(WhileStmt whileStmt) {
        apiList.add("while");
        handleExpression(whileStmt.getCondition());
        handleStmt(whileStmt.getBody());
    }

    private void handleYieldStmt(YieldStmt yieldStmt) {
        apiList.add("yield");
        handleExpression(yieldStmt.getExpression());
    }

    private void handleExpression(Expression expression) {
        if (expression.isArrayAccessExpr())
            handleArrayAccessExpr(expression.toArrayAccessExpr().get());
        else if (expression.isArrayCreationExpr())
            handleArrayCreationExpr(expression.toArrayCreationExpr().get());
        else if (expression.isArrayInitializerExpr())
            handleArrayInitializerExpr(expression.toArrayInitializerExpr().get());
        else if (expression.isAssignExpr())
            handleAssignExpr(expression.toAssignExpr().get());
        else if (expression.isBinaryExpr())
            handleBinaryExpr(expression.toBinaryExpr().get());
        else if (expression.isCastExpr())
            handleCastExpr(expression.toCastExpr().get());
        else if (expression.isConditionalExpr())
            handleConditionalExpr(expression.toConditionalExpr().get());
        else if (expression.isEnclosedExpr())
            handleEnclosedExpr(expression.toEnclosedExpr().get());
        else if (expression.isFieldAccessExpr())
            handleFieldAccessExpr(expression.toFieldAccessExpr().get());
        else if (expression.isInstanceOfExpr())
            handleInstanceOfExpr(expression.toInstanceOfExpr().get());
        else if (expression.isLambdaExpr())
            handleLambdaExpr(expression.toLambdaExpr().get());
        else if (expression.isMethodCallExpr())
            handleMethodCallExpr(expression.toMethodCallExpr().get());
        else if (expression.isMethodReferenceExpr())
            handleMethodReferenceExpr(expression.toMethodReferenceExpr().get());
        else if (expression.isObjectCreationExpr())
            handleObjectCreationExpr(expression.toObjectCreationExpr().get());
        else if (expression.isSwitchExpr())
            handleSwitchExpr(expression.toSwitchExpr().get());
        else if (expression.isUnaryExpr())
            handleUnaryExpr(expression.toUnaryExpr().get());
        else if (expression.isVariableDeclarationExpr())
            handleVariableDeclarationExpr(expression.toVariableDeclarationExpr().get());
        else
            return;
    }

    private void handleArrayAccessExpr(ArrayAccessExpr expr) {
        handleExpression(expr.getName());
        handleExpression(expr.getIndex());
    }

    private void handleArrayCreationExpr(ArrayCreationExpr expr) {
        // 先处理[]中内容
        for (ArrayCreationLevel level : expr.getLevels()) {
            Optional<Expression> dimExpr = level.getDimension();
            if (dimExpr.isPresent())
                handleExpression(dimExpr.get());
        }

        apiList.add("new");
        apiList.add(expr.getElementType().asString() + "Arr" +
                (expr.getLevels().size() > 1 ? "s" : ""));

        // 处理{}中的数组初始化值列表
        Optional<ArrayInitializerExpr> initExpr = expr.getInitializer();
        if (initExpr.isPresent())
            handleArrayInitializerExpr(initExpr.get());
    }

    private void handleArrayInitializerExpr(ArrayInitializerExpr initExpr) {
        for (Expression expression : initExpr.getValues())
            handleExpression(expression);
    }

    private void handleAssignExpr(AssignExpr assignExpr) {
        handleExpression(assignExpr.getValue());
        handleExpression(assignExpr.getTarget());
    }

    private void handleBinaryExpr(BinaryExpr binaryExpr) {
        handleExpression(binaryExpr.getLeft());
        handleExpression(binaryExpr.getRight());
    }

    private void handleCastExpr(CastExpr castExpr) {
        handleExpression(castExpr.getExpression());
    }

    private void handleConditionalExpr(ConditionalExpr conditionalExpr) {
        handleExpression(conditionalExpr.getCondition());
        handleExpression(conditionalExpr.getThenExpr());
        handleExpression(conditionalExpr.getElseExpr());
    }

    private void handleEnclosedExpr(EnclosedExpr enclosedExpr) {
        handleExpression(enclosedExpr.getInner());
    }

    private void handleFieldAccessExpr(FieldAccessExpr fieldAccessExpr) {
        handleExpression(fieldAccessExpr.getScope());
    }

    private void handleInstanceOfExpr(InstanceOfExpr instanceOfExpr) {
        handleExpression(instanceOfExpr.getExpression());
    }

    private void handleLambdaExpr(LambdaExpr lambdaExpr) {
        handleStmt(lambdaExpr.getBody());
    }

    private void handleMethodCallExpr(MethodCallExpr methodCallExpr) {
        // 先处理方法调用中.前的部分
        Optional<Expression> scope = methodCallExpr.getScope();
        if (scope.isPresent()) {
            // 如果.前是简单的变量名或super和this表达式，则将其加入序列
            if (scope.get().isNameExpr())
                apiList.add(scope.get().toNameExpr().get().getNameAsString());
            else if (scope.get().isSuperExpr())
                apiList.add(scope.get().toSuperExpr().get().toString());
            else if (scope.get().isThisExpr())
                apiList.add(scope.get().toThisExpr().get().toString());
            else handleExpression(scope.get());
        }

        // 处理方法调用的参数部分
        for (Expression expression : methodCallExpr.getArguments())
            handleExpression(expression);

        apiList.add(methodCallExpr.getName().asString());
    }

    private void handleMethodReferenceExpr(MethodReferenceExpr methodReferenceExpr) {
        Expression scope = methodReferenceExpr.getScope();
        // 如果.前是简单的变量名，则将其加入序列
        if (scope.isNameExpr())
            apiList.add(scope.toNameExpr().get().getNameAsString());
        else if (scope.isSuperExpr())
            apiList.add(scope.toSuperExpr().get().toString());
        else if (scope.isThisExpr())
            apiList.add(scope.toThisExpr().get().toString());
        else if (scope.isTypeExpr())
            apiList.add(scope.toTypeExpr().get().toString());
        else handleExpression(scope);

        apiList.add(methodReferenceExpr.getIdentifier());
    }

    private void handleObjectCreationExpr(ObjectCreationExpr objectCreationExpr) {
        // 先处理对象创建中.前的部分
        Optional<Expression> scope = objectCreationExpr.getScope();
        if (scope.isPresent()) {
            // 如果.前是简单的变量名，则将其加入序列
            if (scope.get().isNameExpr())
                apiList.add(scope.get().toNameExpr().get().getNameAsString());
            else if (scope.get().isSuperExpr())
                apiList.add(scope.get().toSuperExpr().get().toString());
            else if (scope.get().isThisExpr())
                apiList.add(scope.get().toThisExpr().get().toString());
            else handleExpression(scope.get());
        }

        // 处理方法调用的参数部分
        for (Expression expression : objectCreationExpr.getArguments())
            handleExpression(expression);

        apiList.add("new");
        apiList.add(objectCreationExpr.getTypeAsString());
    }

    private void handleSwitchExpr(SwitchExpr switchExpr) {
        apiList.add("switch");
        handleExpression(switchExpr.getSelector());
        for (SwitchEntry entry : switchExpr.getEntries()) {
            for (Expression labelExpression : entry.getLabels())
                handleExpression(labelExpression);
            for (Statement stmt : entry.getStatements())
                handleStmt(stmt);
        }
    }

    private void handleUnaryExpr(UnaryExpr unaryExpr) {
        handleExpression(unaryExpr.getExpression());
    }

    private void handleVariableDeclarationExpr(VariableDeclarationExpr variableDeclarationExpr) {
        for (VariableDeclarator varDef : variableDeclarationExpr.getVariables()) {
            apiList.add(varDef.getNameAsString());
            Optional<Expression> init = varDef.getInitializer();
            if (init.isPresent())
                handleExpression(init.get());
        }
    }
}
